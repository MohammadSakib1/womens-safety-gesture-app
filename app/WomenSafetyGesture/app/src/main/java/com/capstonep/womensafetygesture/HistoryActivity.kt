package com.capstonep.womensafetygesture

import android.content.Context
import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity

class HistoryActivity : AppCompatActivity() {

    private lateinit var historyContainer: LinearLayout
    private lateinit var tvEmptyHistory: TextView
    private lateinit var btnClearHistory: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_history)

        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Alert History"

        historyContainer = findViewById(R.id.historyContainer)
        tvEmptyHistory = findViewById(R.id.tvEmptyHistory)
        btnClearHistory = findViewById(R.id.btnClearHistory)

        btnClearHistory.setOnClickListener {
            confirmClearHistory()
        }

        loadHistory()
    }

    private fun loadHistory() {
        val historyPrefs = getSharedPreferences("AlertHistory", Context.MODE_PRIVATE)
        val count = historyPrefs.getInt("count", 0)

        historyContainer.removeAllViews()

        if (count == 0) {
            tvEmptyHistory.visibility = android.view.View.VISIBLE
            return
        }

        tvEmptyHistory.visibility = android.view.View.GONE

        // Show newest first
        for (i in count - 1 downTo 0) {
            val type = historyPrefs.getString("alert_${i}_type", "") ?: ""
            val time = historyPrefs.getString("alert_${i}_time", "") ?: ""
            val contact = historyPrefs.getString("alert_${i}_contact", "") ?: ""
            val status = historyPrefs.getString("alert_${i}_status", "") ?: ""

            val card = createHistoryCard(type, time, contact, status)
            historyContainer.addView(card)
        }
    }

    private fun createHistoryCard(
        type: String,
        time: String,
        contact: String,
        status: String
    ): LinearLayout {
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#16213E"))
            setPadding(32, 24, 32, 24)
            val params = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            params.setMargins(0, 0, 0, 16)
            layoutParams = params
        }

        // Type
        val typeView = TextView(this).apply {
            text = type
            textSize = 18f
            setTextColor(Color.WHITE)
            typeface = android.graphics.Typeface.DEFAULT_BOLD
        }

        // Time
        val timeView = TextView(this).apply {
            text = "🕐 $time"
            textSize = 13f
            setTextColor(Color.parseColor("#AAAAAA"))
            setPadding(0, 8, 0, 4)
        }

        // Contact
        val contactView = TextView(this).apply {
            text = "👤 $contact"
            textSize = 13f
            setTextColor(Color.parseColor("#AAAAAA"))
            setPadding(0, 0, 0, 4)
        }

        // Status
        val statusView = TextView(this).apply {
            text = status
            textSize = 13f
            setTextColor(Color.parseColor("#4CAF50"))
        }

        card.addView(typeView)
        card.addView(timeView)
        card.addView(contactView)
        card.addView(statusView)

        return card
    }

    private fun confirmClearHistory() {
        AlertDialog.Builder(this)
            .setTitle("Clear History")
            .setMessage("Are you sure you want to delete all alert history?")
            .setPositiveButton("Clear") { _, _ ->
                clearHistory()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun clearHistory() {
        val historyPrefs = getSharedPreferences("AlertHistory", Context.MODE_PRIVATE)
        historyPrefs.edit().clear().apply()
        Toast.makeText(this, "History cleared!", Toast.LENGTH_SHORT).show()
        loadHistory()
    }

    override fun onSupportNavigateUp(): Boolean {
        onBackPressedDispatcher.onBackPressed()
        return true
    }
}