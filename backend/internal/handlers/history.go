package handlers

import (
	"encoding/json"
	"math-solver-backend/internal/models"
	"strconv"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
)

type HistoryHandler struct {
	db *gorm.DB
}

func NewHistoryHandler(db *gorm.DB) *HistoryHandler {
	return &HistoryHandler{db: db}
}

func (h *HistoryHandler) GetHistory(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(401, gin.H{"error": "User not authenticated"})
		return
	}

	// Get user ID from query parameter (for now, using authenticated user)
	queryUserID := c.Query("user_id")
	if queryUserID != "" {
		if parsedUserID, err := strconv.ParseUint(queryUserID, 10, 32); err == nil {
			userID = uint(parsedUserID)
		}
	}

	var equations []models.Equation
	if err := h.db.Where("user_id = ?", userID).Order("created_at DESC").Find(&equations).Error; err != nil {
		c.JSON(500, gin.H{"error": "Failed to fetch history"})
		return
	}

	var historyItems []models.HistoryItem
	for _, eq := range equations {
		// Get solution for this equation
		var solution models.Solution
		var steps, solutionText []string

		if err := h.db.Where("equation_id = ?", eq.ID).First(&solution).Error; err == nil {
			json.Unmarshal([]byte(solution.Steps), &steps)
			json.Unmarshal([]byte(solution.Solution), &solutionText)
		}

		historyItems = append(historyItems, models.HistoryItem{
			ID:         eq.ID,
			Equation:   eq.Input,
			Solution:   solutionText,
			Timestamp:  eq.CreatedAt,
			Confidence: eq.Confidence,
		})
	}

	c.JSON(200, historyItems)
}

func (h *HistoryHandler) DeleteHistoryItem(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(401, gin.H{"error": "User not authenticated"})
		return
	}

	equationID := c.Param("id")
	if equationID == "" {
		c.JSON(400, gin.H{"error": "Equation ID required"})
		return
	}

	// Check if equation belongs to user
	var equation models.Equation
	if err := h.db.Where("id = ? AND user_id = ?", equationID, userID).First(&equation).Error; err != nil {
		c.JSON(404, gin.H{"error": "Equation not found"})
		return
	}

	// Delete equation (cascade will delete solution)
	if err := h.db.Delete(&equation).Error; err != nil {
		c.JSON(500, gin.H{"error": "Failed to delete equation"})
		return
	}

	c.JSON(200, gin.H{"message": "Equation deleted successfully"})
}
