# syntax=docker/dockerfile:1
FROM golang:1.22-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY go_backend ./go_backend
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/tensorshadow ./go_backend

FROM gcr.io/distroless/static-debian12:nonroot
WORKDIR /app
COPY --from=build /out/tensorshadow /app/tensorshadow
COPY configs/config.yaml /app/configs/config.yaml
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/app/tensorshadow"]
