package main

import (
	"log"
	"math-solver-backend/internal/config"
	"math-solver-backend/internal/database"
	"math-solver-backend/internal/handlers"
	"math-solver-backend/internal/middleware"
	"math-solver-backend/internal/models"

	"github.com/gin-gonic/gin"
)

func main() {
	// Load configuration
	cfg := config.Load()

	// Initialize database
	db, err := database.Initialize(cfg.DatabaseURL)
	if err != nil {
		log.Fatal("Failed to connect to database:", err)
	}

	// Auto-migrate models
	if err := db.AutoMigrate(&models.User{}, &models.Equation{}, &models.Solution{}); err != nil {
		log.Fatal("Failed to migrate database:", err)
	}

	// Initialize handlers
	equationHandler := handlers.NewEquationHandler(db, cfg.AIServiceURL)
	authHandler := handlers.NewAuthHandler(db, cfg.JWTSecret)
	historyHandler := handlers.NewHistoryHandler(db)

	// Setup router
	router := gin.Default()

	// Middleware
	router.Use(middleware.CORS())
	router.Use(middleware.Logger())

	// Health check
	router.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})

	// API routes
	api := router.Group("/api")
	{
		// Authentication routes
		auth := api.Group("/auth")
		{
			auth.POST("/register", authHandler.Register)
			auth.POST("/login", authHandler.Login)
		}

		// Public routes (for demo)
		api.POST("/solve", equationHandler.SolveEquation)

		// Protected routes
		protected := api.Group("/")
		protected.Use(middleware.AuthRequired(cfg.JWTSecret))
		{
			// History
			protected.GET("/history", historyHandler.GetHistory)
			protected.DELETE("/history/:id", historyHandler.DeleteHistoryItem)
		}
	}

	// Start server
	log.Printf("Server starting on port %s", cfg.Port)
	if err := router.Run(":" + cfg.Port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}
