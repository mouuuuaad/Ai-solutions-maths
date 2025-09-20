package handlers

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"

	"math-solver-backend/internal/models"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
)

type EquationHandler struct {
	db           *gorm.DB
	aiServiceURL string
}

func NewEquationHandler(db *gorm.DB, aiServiceURL string) *EquationHandler {
	return &EquationHandler{
		db:           db,
		aiServiceURL: aiServiceURL,
	}
}

func (h *EquationHandler) SolveEquation(c *gin.Context) {
	var req models.SolveEquationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// For demo purposes, use a default user ID
	userID := uint(1) // Demo user ID

	// Forward request to AI service
	aiReq := map[string]string{
		"equation_text": req.EquationText,
	}

	jsonData, err := json.Marshal(aiReq)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to prepare request"})
		return
	}

	resp, err := http.Post(h.aiServiceURL+"/solve", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "AI service unavailable"})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "AI service error"})
		return
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read AI response"})
		return
	}

	var aiResponse models.SolveEquationResponse
	if err := json.Unmarshal(body, &aiResponse); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse AI response"})
		return
	}

	// Save equation and solution to database
	equation := models.Equation{
		UserID:     userID,
		Input:      aiResponse.Input,
		Normalized: aiResponse.Normalized,
		ImageData:  "", // No image data for text input
		Confidence: aiResponse.Confidence,
	}

	if err := h.db.Create(&equation).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save equation"})
		return
	}

	// Convert steps and solution to JSON strings
	stepsJSON, _ := json.Marshal(aiResponse.Steps)
	solutionJSON, _ := json.Marshal(aiResponse.Solution)

	solution := models.Solution{
		EquationID: equation.ID,
		Steps:      string(stepsJSON),
		Solution:   string(solutionJSON),
	}

	if err := h.db.Create(&solution).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save solution"})
		return
	}

	c.JSON(http.StatusOK, aiResponse)
}
