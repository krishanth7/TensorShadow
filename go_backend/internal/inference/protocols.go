package inference

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"
)

const maxResponseBytes = 32 << 20

func doJSON(ctx context.Context, client *http.Client, method, u string, in, out any) error {
	var body io.Reader
	if in != nil {
		b, err := json.Marshal(in)
		if err != nil {
			return err
		}
		body = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, u, body)
	if err != nil {
		return err
	}
	if in != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return err
	}
	if resp.StatusCode >= 300 {
		return &remoteError{status: resp.StatusCode, msg: errorMessage(raw)}
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("decode model server response: %w", err)
	}
	return nil
}

// errorMessage extracts {"error": "..."} (both protocols) or returns the body.
func errorMessage(raw []byte) string {
	var e struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(raw, &e) == nil && e.Error != "" {
		return e.Error
	}
	s := strings.TrimSpace(string(raw))
	if len(s) > 300 {
		s = s[:300] + "…"
	}
	return s
}

// ---------------------------------------------------------------------------
// TensorFlow Serving REST API
// https://www.tensorflow.org/tfx/serving/api_rest
// ---------------------------------------------------------------------------

type tfServing struct {
	client *http.Client
	base   string
	cfg    Config
}

func (t *tfServing) modelURL() string {
	u := t.base + "/v1/models/" + url.PathEscape(t.cfg.Model)
	if t.cfg.Version != "" {
		u += "/versions/" + url.PathEscape(t.cfg.Version)
	}
	return u
}

func (t *tfServing) predict(ctx context.Context, instances [][]float64) ([][]float64, string, error) {
	var resp struct {
		Predictions []json.RawMessage `json:"predictions"`
	}
	req := map[string]any{"signature_name": "serving_default", "instances": instances}
	if err := doJSON(ctx, t.client, http.MethodPost, t.modelURL()+":predict", req, &resp); err != nil {
		return nil, "", err
	}
	out := make([][]float64, len(resp.Predictions))
	for i, p := range resp.Predictions {
		v, err := decodePrediction(p, t.cfg.OutputName)
		if err != nil {
			return nil, "", fmt.Errorf("prediction %d: %w", i, err)
		}
		out[i] = v
	}
	return out, t.cfg.Version, nil
}

// decodePrediction accepts a scalar, a (nested) array, or — for multi-output
// signatures — an object of named outputs.
func decodePrediction(raw json.RawMessage, outputName string) ([]float64, error) {
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) == nil {
		if outputName != "" {
			v, ok := obj[outputName]
			if !ok {
				return nil, fmt.Errorf("output %q not in prediction (have %v)", outputName, keys(obj))
			}
			return flatten(v)
		}
		if len(obj) == 1 {
			for _, v := range obj {
				return flatten(v)
			}
		}
		return nil, fmt.Errorf("multi-output signature: set inference.output_name to one of %v", keys(obj))
	}
	return flatten(raw)
}

func flatten(raw json.RawMessage) ([]float64, error) {
	var f float64
	if json.Unmarshal(raw, &f) == nil {
		return []float64{f}, nil
	}
	var arr []json.RawMessage
	if err := json.Unmarshal(raw, &arr); err != nil {
		return nil, fmt.Errorf("prediction is not numeric: %s", string(raw))
	}
	var out []float64
	for _, a := range arr {
		v, err := flatten(a)
		if err != nil {
			return nil, err
		}
		out = append(out, v...)
	}
	return out, nil
}

func keys(m map[string]json.RawMessage) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func (t *tfServing) status(ctx context.Context) Status {
	var resp struct {
		ModelVersionStatus []struct {
			Version string `json:"version"`
			State   string `json:"state"`
			Status  struct {
				ErrorCode    string `json:"error_code"`
				ErrorMessage string `json:"error_message"`
			} `json:"status"`
		} `json:"model_version_status"`
	}
	if err := doJSON(ctx, t.client, http.MethodGet, t.modelURL(), nil, &resp); err != nil {
		return Status{State: "UNREACHABLE", Detail: err.Error()}
	}
	st := Status{State: "UNKNOWN"}
	for _, v := range resp.ModelVersionStatus {
		st.Versions = append(st.Versions, v.Version+":"+v.State)
		if v.State == "AVAILABLE" {
			st.Ready, st.State = true, "AVAILABLE"
		} else if !st.Ready {
			st.State, st.Detail = v.State, v.Status.ErrorMessage
		}
	}
	return st
}

// ---------------------------------------------------------------------------
// Open Inference Protocol v2 (KServe v2 / Triton) — ONNX Runtime models
// https://kserve.github.io/website/latest/modelserving/data_plane/v2_protocol/
// ---------------------------------------------------------------------------

type openInference struct {
	client *http.Client
	base   string
	cfg    Config

	mu        sync.Mutex
	inputName string // discovered from model metadata when not configured
}

type v2Tensor struct {
	Name     string    `json:"name"`
	Shape    []int     `json:"shape"`
	Datatype string    `json:"datatype"`
	Data     []float64 `json:"data"`
}

type v2Metadata struct {
	Name     string   `json:"name"`
	Versions []string `json:"versions"`
	Platform string   `json:"platform"`
	Inputs   []struct {
		Name     string `json:"name"`
		Datatype string `json:"datatype"`
		Shape    []int  `json:"shape"`
	} `json:"inputs"`
	Outputs []struct {
		Name     string `json:"name"`
		Datatype string `json:"datatype"`
		Shape    []int  `json:"shape"`
	} `json:"outputs"`
}

func (o *openInference) modelURL() string {
	u := o.base + "/v2/models/" + url.PathEscape(o.cfg.Model)
	if o.cfg.Version != "" {
		u += "/versions/" + url.PathEscape(o.cfg.Version)
	}
	return u
}

func (o *openInference) resolveInput(ctx context.Context) (string, error) {
	if o.cfg.InputName != "" {
		return o.cfg.InputName, nil
	}
	o.mu.Lock()
	name := o.inputName
	o.mu.Unlock()
	if name != "" {
		return name, nil
	}
	var md v2Metadata
	if err := doJSON(ctx, o.client, http.MethodGet, o.modelURL(), nil, &md); err != nil {
		return "", fmt.Errorf("discover input name from model metadata: %w", err)
	}
	if len(md.Inputs) != 1 {
		return "", fmt.Errorf("model has %d inputs; set inference.input_name", len(md.Inputs))
	}
	o.mu.Lock()
	o.inputName = md.Inputs[0].Name
	o.mu.Unlock()
	return md.Inputs[0].Name, nil
}

func (o *openInference) predict(ctx context.Context, instances [][]float64) ([][]float64, string, error) {
	inName, err := o.resolveInput(ctx)
	if err != nil {
		return nil, "", err
	}
	n, d := len(instances), len(instances[0])
	flat := make([]float64, 0, n*d)
	for _, inst := range instances {
		flat = append(flat, inst...)
	}
	req := map[string]any{"inputs": []v2Tensor{{Name: inName, Shape: []int{n, d}, Datatype: "FP32", Data: flat}}}
	if o.cfg.OutputName != "" {
		req["outputs"] = []map[string]string{{"name": o.cfg.OutputName}}
	}
	var resp struct {
		ModelVersion string     `json:"model_version"`
		Outputs      []v2Tensor `json:"outputs"`
	}
	if err := doJSON(ctx, o.client, http.MethodPost, o.modelURL()+"/infer", req, &resp); err != nil {
		return nil, "", err
	}
	var out *v2Tensor
	for i := range resp.Outputs {
		if o.cfg.OutputName == "" || resp.Outputs[i].Name == o.cfg.OutputName {
			out = &resp.Outputs[i]
			break
		}
	}
	if out == nil {
		return nil, "", fmt.Errorf("model server response has no output %q", o.cfg.OutputName)
	}
	if len(out.Shape) == 0 || out.Shape[0] != n {
		return nil, "", fmt.Errorf("output %q has shape %v, want batch dimension %d", out.Name, out.Shape, n)
	}
	per := len(out.Data) / n
	if per == 0 || per*n != len(out.Data) {
		return nil, "", fmt.Errorf("output %q has %d values for batch %d", out.Name, len(out.Data), n)
	}
	res := make([][]float64, n)
	for i := range res {
		res[i] = out.Data[i*per : (i+1)*per]
	}
	return res, resp.ModelVersion, nil
}

func (o *openInference) status(ctx context.Context) Status {
	if err := doJSON(ctx, o.client, http.MethodGet, o.modelURL()+"/ready", nil, nil); err != nil {
		return Status{State: "UNAVAILABLE", Detail: err.Error()}
	}
	st := Status{Ready: true, State: "AVAILABLE"}
	var md v2Metadata
	if err := doJSON(ctx, o.client, http.MethodGet, o.modelURL(), nil, &md); err == nil {
		st.Versions = md.Versions
		st.Metadata = map[string]any{"platform": md.Platform, "inputs": md.Inputs, "outputs": md.Outputs}
	}
	return st
}
