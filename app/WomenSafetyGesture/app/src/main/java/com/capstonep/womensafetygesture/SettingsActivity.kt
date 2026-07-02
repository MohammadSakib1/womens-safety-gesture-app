package com.capstonep.womensafetygesture

import android.content.Context
import android.content.SharedPreferences
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

class SettingsActivity : AppCompatActivity() {

    private lateinit var etContactName: EditText
    private lateinit var etPhoneNumber: EditText
    private lateinit var etSmsMessage: EditText
    private lateinit var seekBarThreshold: SeekBar
    private lateinit var tvThresholdValue: TextView
    private lateinit var btnSave: Button
    private lateinit var btnTestAlert: Button
    private lateinit var prefs: SharedPreferences

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        // Setup back button
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Emergency Settings"

        prefs = getSharedPreferences("WomenSafetyPrefs", Context.MODE_PRIVATE)

        etContactName = findViewById(R.id.etContactName)
        etPhoneNumber = findViewById(R.id.etPhoneNumber)
        etSmsMessage = findViewById(R.id.etSmsMessage)
        seekBarThreshold = findViewById(R.id.seekBarThreshold)
        tvThresholdValue = findViewById(R.id.tvThresholdValue)
        btnSave = findViewById(R.id.btnSave)
        btnTestAlert = findViewById(R.id.btnTestAlert)

        loadSettings()

        seekBarThreshold.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val threshold = 50 + progress // range: 50% to 90%
                tvThresholdValue.text = "Threshold: $threshold%"
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        btnSave.setOnClickListener {
            saveSettings()
        }

        btnTestAlert.setOnClickListener {
            testAlert()
        }
    }

    private fun loadSettings() {
        etContactName.setText(prefs.getString("contact_name", ""))
        etPhoneNumber.setText(prefs.getString("phone_number", ""))
        etSmsMessage.setText(prefs.getString("sms_message",
            "EMERGENCY! I need help! Please contact me immediately."))
        val threshold = prefs.getInt("confidence_threshold", 78)
        val progress = threshold - 50
        seekBarThreshold.progress = progress
        tvThresholdValue.text = "Threshold: $threshold%"
    }

    private fun saveSettings() {
        val name = etContactName.text.toString().trim()
        val phone = etPhoneNumber.text.toString().trim()
        val message = etSmsMessage.text.toString().trim()
        val threshold = 50 + seekBarThreshold.progress

        if (phone.isEmpty()) {
            Toast.makeText(this, "Please enter emergency phone number!", Toast.LENGTH_SHORT).show()
            return
        }

        prefs.edit()
            .putString("contact_name", name)
            .putString("phone_number", phone)
            .putString("sms_message", message)
            .putInt("confidence_threshold", threshold)
            .apply()

        Toast.makeText(this,
            "Settings saved! Emergency contact: $name ($phone)",
            Toast.LENGTH_LONG).show()
    }

    private fun testAlert() {
        val phone = etPhoneNumber.text.toString().trim()
        if (phone.isEmpty()) {
            Toast.makeText(this, "Please enter a phone number first!", Toast.LENGTH_SHORT).show()
            return
        }
        val alertManager = AlertManager(this)
        alertManager.triggerAlert()
        Toast.makeText(this, "Test alert sent to $phone!", Toast.LENGTH_LONG).show()
    }

    override fun onSupportNavigateUp(): Boolean {
        onBackPressedDispatcher.onBackPressed()
        return true
    }
}