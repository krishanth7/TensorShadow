package main

import (
	"log"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
	"github.com/TensorShadow/TensorShadow/go_backend/handlers"
	"github.com/spf13/viper"
)

func main() {
	// Initialize Configuration
	initConfig()

	// Set up Gin router
	router := gin.Default()

	// API Routes
	v1 := router.Group("/api/v1")
	{
		v1.GET("/health", handlers.HealthCheck)
		v1.POST("/predict", handlers.HandlePrediction)
		v1.GET("/stats", handlers.GetAnalytics)
	}

	port := viper.GetString("server.port")
	if port == "" {
		port = "8080"
	}

	log.Printf("TensorShadow Backend API running on port %s", port)
	if err := router.Run(":" + port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

func initConfig() {
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath("./configs")
	viper.AddConfigPath(".")

	if err := viper.ReadInConfig(); err != nil {
		log.Println("No config file found, using defaults")
	}

	viper.SetDefault("server.port", "8080")
	viper.SetDefault("env", "development")
}

