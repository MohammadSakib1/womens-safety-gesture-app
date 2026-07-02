package com.capstonep.womensafetygesture

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Location
import android.net.Uri
import android.telephony.SmsManager
import android.util.Log
import androidx.core.content.ContextCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority

class AlertManager(private val context: Context) {

    private val fusedLocationClient: FusedLocationProviderClient =
        LocationServices.getFusedLocationProviderClient(context)

    private var alertSent = false
    private var lastAlertTime = 0L
    private val alertCooldown = 10000L

    private fun getEmergencyNumber(): String {
        val prefs = context.getSharedPreferences("WomenSafetyPrefs", Context.MODE_PRIVATE)
        return prefs.getString("phone_number", "") ?: ""
    }

    private fun getSmsMessage(): String {
        val prefs = context.getSharedPreferences("WomenSafetyPrefs", Context.MODE_PRIVATE)
        return prefs.getString("sms_message",
            "EMERGENCY! I need help! Please contact me immediately.") ?: ""
    }

    private fun getConfidenceThreshold(): Float {
        val prefs = context.getSharedPreferences("WomenSafetyPrefs", Context.MODE_PRIVATE)
        return prefs.getInt("confidence_threshold", 78) / 100f
    }

    fun getThreshold(): Float = getConfidenceThreshold()

    fun triggerAlert() {
        val emergencyNumber = getEmergencyNumber()
        if (emergencyNumber.isEmpty()) {
            Log.w("AlertManager", "No emergency number set! Go to Settings.")
            return
        }

        val currentTime = System.currentTimeMillis()
        if (alertSent && (currentTime - lastAlertTime) < alertCooldown) return

        alertSent = true
        lastAlertTime = currentTime

        getLocationAndSendAlert()
        saveToHistory("🆘 HELP Alert", "SMS + Call + GPS sent ✅")
    }

    private fun getLocationAndSendAlert() {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
            == PackageManager.PERMISSION_GRANTED) {
            fusedLocationClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null)
                .addOnSuccessListener { location: Location? ->
                    val locationText = if (location != null) {
                        "https://maps.google.com/?q=${location.latitude},${location.longitude}"
                    } else {
                        "Location unavailable"
                    }
                    sendSmsAlert(locationText)
                    makeEmergencyCall()
                }
                .addOnFailureListener {
                    sendSmsAlert("Location unavailable")
                    makeEmergencyCall()
                }
        } else {
            sendSmsAlert("Location unavailable")
            makeEmergencyCall()
        }
    }

    private fun sendSmsAlert(locationUrl: String) {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.SEND_SMS)
            != PackageManager.PERMISSION_GRANTED) return

        val emergencyNumber = getEmergencyNumber()
        if (emergencyNumber.isEmpty()) return

        try {
            val baseMessage = getSmsMessage()
            val fullMessage = "$baseMessage\nMy location: $locationUrl"
            val smsManager = context.getSystemService(SmsManager::class.java)
            smsManager.sendTextMessage(emergencyNumber, null, fullMessage, null, null)
            Log.d("AlertManager", "SMS sent to $emergencyNumber")
        } catch (e: Exception) {
            Log.e("AlertManager", "SMS failed: ${e.message}")
        }
    }

    private fun makeEmergencyCall() {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.CALL_PHONE)
            != PackageManager.PERMISSION_GRANTED) return

        val emergencyNumber = getEmergencyNumber()
        if (emergencyNumber.isEmpty()) return

        try {
            val callIntent = Intent(Intent.ACTION_CALL).apply {
                data = Uri.parse("tel:$emergencyNumber")
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            context.startActivity(callIntent)
        } catch (e: Exception) {
            Log.e("AlertManager", "Call failed: ${e.message}")
        }
    }
    fun triggerCall() {
        val currentTime = System.currentTimeMillis()
        if (alertSent && (currentTime - lastAlertTime) < alertCooldown) return
        alertSent = true
        lastAlertTime = currentTime
        makeEmergencyCall()
        saveToHistory("📞 CALL Alert", "Emergency call made ✅")
    }

    fun cancelAlert() {
        alertSent = false
        lastAlertTime = 0L
        Log.d("AlertManager", "Alert cancelled by STOP gesture")
    }

    fun sendSafeMessage() {
        val emergencyNumber = getEmergencyNumber()
        if (emergencyNumber.isEmpty()) return

        val currentTime = System.currentTimeMillis()
        if (alertSent && (currentTime - lastAlertTime) < alertCooldown) return
        alertSent = true
        lastAlertTime = currentTime

        try {
            val message = "I AM SAFE now. Please don't worry. - Sent via Women Safety App"
            val smsManager = context.getSystemService(SmsManager::class.java)
            smsManager.sendTextMessage(emergencyNumber, null, message, null, null)
            saveToHistory("✅ SAFE Message", "I am safe SMS sent ✅")
            Log.d("AlertManager", "Safe message sent to $emergencyNumber")
        } catch (e: Exception) {
            Log.e("AlertManager", "Safe SMS failed: ${e.message}")
        }
    }
    private fun saveToHistory(type: String, status: String) {
        val prefs = context.getSharedPreferences("WomenSafetyPrefs", Context.MODE_PRIVATE)
        val contactName = prefs.getString("contact_name", "Emergency Contact") ?: "Emergency Contact"
        val phoneNumber = getEmergencyNumber()

        val timestamp = java.text.SimpleDateFormat(
            "dd MMM yyyy, hh:mm a",
            java.util.Locale.getDefault()
        ).format(java.util.Date())

        val historyPrefs = context.getSharedPreferences("AlertHistory", Context.MODE_PRIVATE)
        val count = historyPrefs.getInt("count", 0)

        historyPrefs.edit()
            .putString("alert_${count}_type", type)
            .putString("alert_${count}_time", timestamp)
            .putString("alert_${count}_contact", "$contactName ($phoneNumber)")
            .putString("alert_${count}_status", status)
            .putInt("count", count + 1)
            .apply()
    }
    fun resetAlert() {
        alertSent = false
    }
}