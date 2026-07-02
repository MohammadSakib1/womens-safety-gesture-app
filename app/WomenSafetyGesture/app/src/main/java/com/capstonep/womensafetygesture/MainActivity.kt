package com.capstonep.womensafetygesture

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.os.Bundle
import android.os.CountDownTimer
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var gestureLabel: TextView
    private lateinit var confidenceLabel: TextView
    private lateinit var tvCountdown: TextView
    private lateinit var countdownLayout: LinearLayout
    private lateinit var btnCancelAlert: Button
    private lateinit var cameraExecutor: ExecutorService
    private var handLandmarker: HandLandmarker? = null
    private var gestureClassifier: GestureClassifier? = null
    private var alertManager: AlertManager? = null
    private var countDownTimer: CountDownTimer? = null
    private var isCountingDown = false

    private val requestMultiplePermissions =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            if (permissions[Manifest.permission.CAMERA] == true) {
                startCamera()
            } else {
                gestureLabel.text = "Camera permission denied"
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        gestureLabel = findViewById(R.id.gestureLabel)
        confidenceLabel = findViewById(R.id.confidenceLabel)
        tvCountdown = findViewById(R.id.tvCountdown)
        countdownLayout = findViewById(R.id.countdownLayout)
        btnCancelAlert = findViewById(R.id.btnCancelAlert)

        cameraExecutor = Executors.newSingleThreadExecutor()
        gestureClassifier = GestureClassifier(this)
        alertManager = AlertManager(this)

        findViewById<ImageButton>(R.id.btnSettings).setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }
        findViewById<ImageButton>(R.id.btnHistory).setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java))
        }

        btnCancelAlert.setOnClickListener {
            cancelCountdown()
        }

        setupMediaPipe()
        requestPermissions()
    }

    override fun onResume() {
        super.onResume()
        val prefs = getSharedPreferences("WomenSafetyPrefs", Context.MODE_PRIVATE)
        val phone = prefs.getString("phone_number", "") ?: ""
        if (phone.isEmpty()) {
            gestureLabel.text = "⚠️ Set emergency contact in Settings!"
            gestureLabel.setTextColor(Color.YELLOW)
        }
    }

    private fun startHelpCountdown() {
        if (isCountingDown) return
        isCountingDown = true
        countdownLayout.visibility = View.VISIBLE

        countDownTimer = object : CountDownTimer(3000, 1000) {
            override fun onTick(millisUntilFinished: Long) {
                val secondsLeft = (millisUntilFinished / 1000) + 1
                runOnUiThread {
                    tvCountdown.text = secondsLeft.toString()
                }
            }
            override fun onFinish() {
                runOnUiThread {
                    countdownLayout.visibility = View.GONE
                    isCountingDown = false
                    alertManager?.triggerAlert()
                }
            }
        }.start()
    }

    private fun cancelCountdown() {
        countDownTimer?.cancel()
        countDownTimer = null
        isCountingDown = false
        countdownLayout.visibility = View.GONE
        gestureLabel.text = "Alert Cancelled"
        gestureLabel.setTextColor(Color.YELLOW)
    }

    private fun requestPermissions() {
        val permissions = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.SEND_SMS,
            Manifest.permission.CALL_PHONE,
            Manifest.permission.ACCESS_FINE_LOCATION
        )
        val allGranted = permissions.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
        if (allGranted) startCamera()
        else requestMultiplePermissions.launch(permissions)
    }

    private fun setupMediaPipe() {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build()
            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setNumHands(1)
                .setRunningMode(RunningMode.IMAGE)
                .build()
            handLandmarker = HandLandmarker.createFromOptions(this, options)
        } catch (e: Exception) {
            Log.e("MainActivity", "MediaPipe setup failed: ${e.message}")
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build().also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImage(imageProxy)
                    }
                }
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    imageAnalyzer
                )
            } catch (e: Exception) {
                Log.e("MainActivity", "Camera binding failed: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImage(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap()
        val matrix = Matrix().apply {
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
        }
        val rotatedBitmap = Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
        )
        val mpImage = BitmapImageBuilder(rotatedBitmap).build()

        try {
            val result: HandLandmarkerResult? = handLandmarker?.detect(mpImage)
            if (result != null && result.landmarks().isNotEmpty()) {
                val landmarks = result.landmarks()[0]
                val (label, confidence) = gestureClassifier!!.classify(landmarks)
                val threshold = alertManager?.getThreshold() ?: 0.78f

                runOnUiThread {
                    if (!isCountingDown) {
                        confidenceLabel.text = "Confidence: ${"%.1f".format(confidence * 100)}%"
                        when {
                            label == "help" && confidence > threshold -> {
                                gestureLabel.text = "🆘 HELP DETECTED!"
                                gestureLabel.setTextColor(Color.RED)
                                startHelpCountdown()
                            }
                            label == "call" && confidence > threshold -> {
                                gestureLabel.text = "📞 CALLING..."
                                gestureLabel.setTextColor(Color.GREEN)
                                alertManager?.triggerCall()
                            }
                            label == "stop" && confidence > threshold -> {
                                gestureLabel.text = "🛑 ALERT CANCELLED"
                                gestureLabel.setTextColor(Color.YELLOW)
                                alertManager?.cancelAlert()
                            }
                            label == "ok" && confidence > threshold -> {
                                gestureLabel.text = "✅ I AM SAFE"
                                gestureLabel.setTextColor(Color.GREEN)
                                alertManager?.sendSafeMessage()
                            }
                            label == "mute" && confidence > threshold -> {
                                gestureLabel.text = "🔇 MUTED"
                                gestureLabel.setTextColor(Color.GRAY)
                            }
                            else -> {
                                gestureLabel.text = label.uppercase()
                                gestureLabel.setTextColor(Color.WHITE)
                                alertManager?.resetAlert()
                            }
                        }
                    }
                }
            } else {
                runOnUiThread {
                    if (!isCountingDown) {
                        gestureLabel.text = "No hand detected"
                        gestureLabel.setTextColor(Color.WHITE)
                        confidenceLabel.text = ""
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Detection error: ${e.message}")
        }
        imageProxy.close()
    }

    override fun onDestroy() {
        super.onDestroy()
        countDownTimer?.cancel()
        cameraExecutor.shutdown()
        handLandmarker?.close()
        gestureClassifier?.close()
    }
}