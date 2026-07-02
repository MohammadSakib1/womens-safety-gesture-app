package com.capstonep.womensafetygesture

import android.content.Context
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.acos
import kotlin.math.sqrt

class GestureClassifier(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()

    // Landmark indices
    private val WRIST = 0
    private val THUMB_CMC = 1
    private val THUMB_MCP = 2
    private val THUMB_IP = 3
    private val THUMB_TIP = 4
    private val INDEX_MCP = 5
    private val INDEX_PIP = 6
    private val INDEX_DIP = 7
    private val INDEX_TIP = 8
    private val MIDDLE_MCP = 9
    private val MIDDLE_PIP = 10
    private val MIDDLE_DIP = 11
    private val MIDDLE_TIP = 12
    private val RING_MCP = 13
    private val RING_PIP = 14
    private val RING_DIP = 15
    private val RING_TIP = 16
    private val PINKY_MCP = 17
    private val PINKY_PIP = 18
    private val PINKY_DIP = 19
    private val PINKY_TIP = 20

    private val DISTANCE_PAIRS = listOf(
        Pair(WRIST, THUMB_TIP), Pair(WRIST, INDEX_TIP),
        Pair(WRIST, MIDDLE_TIP), Pair(WRIST, RING_TIP),
        Pair(WRIST, PINKY_TIP), Pair(THUMB_TIP, INDEX_TIP),
        Pair(INDEX_TIP, MIDDLE_TIP), Pair(MIDDLE_TIP, RING_TIP),
        Pair(RING_TIP, PINKY_TIP), Pair(THUMB_TIP, PINKY_TIP),
        Pair(INDEX_MCP, PINKY_MCP), Pair(THUMB_CMC, PINKY_MCP)
    )

    private val ANGLE_TRIPLETS = listOf(
        Triple(THUMB_MCP, THUMB_IP, THUMB_TIP),
        Triple(INDEX_MCP, INDEX_PIP, INDEX_TIP),
        Triple(MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
        Triple(RING_MCP, RING_PIP, RING_TIP),
        Triple(PINKY_MCP, PINKY_PIP, PINKY_TIP),
        Triple(WRIST, INDEX_MCP, INDEX_TIP),
        Triple(WRIST, MIDDLE_MCP, MIDDLE_TIP),
        Triple(WRIST, RING_MCP, RING_TIP),
        Triple(WRIST, PINKY_MCP, PINKY_TIP)
    )

    init {
        loadModel()
        loadLabels()
    }

    private fun loadModel() {
        try {
            interpreter = Interpreter(loadModelFile())
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd("model.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    private fun loadLabels() {
        try {
            labels = context.assets.open("labels.txt")
                .bufferedReader()
                .readLines()
                .filter { it.isNotBlank() }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun classify(landmarks: List<NormalizedLandmark>): Pair<String, Float> {
        if (interpreter == null || labels.isEmpty()) return Pair("unknown", 0f)

        val input = buildFeatureVector(landmarks)
        val output = Array(1) { FloatArray(labels.size) }
        interpreter!!.run(arrayOf(input), output)

        val probabilities = output[0]
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val confidence = probabilities[maxIndex]
        val label = if (maxIndex < labels.size) labels[maxIndex] else "unknown"

        return Pair(label, confidence)
    }

    private fun buildFeatureVector(landmarks: List<NormalizedLandmark>): FloatArray {
        // Step 1: Extract raw coords
        val coords = Array(21) { i ->
            floatArrayOf(landmarks[i].x(), landmarks[i].y(), landmarks[i].z())
        }

        // Step 2: Normalize — subtract wrist, divide by palm size
        val wrist = coords[WRIST].copyOf()
        for (i in coords.indices) {
            coords[i][0] -= wrist[0]
            coords[i][1] -= wrist[1]
            coords[i][2] -= wrist[2]
        }

        val palmSize = distance(coords[MIDDLE_MCP], coords[WRIST])
        if (palmSize > 1e-6f) {
            for (i in coords.indices) {
                coords[i][0] /= palmSize
                coords[i][1] /= palmSize
                coords[i][2] /= palmSize
            }
        }

        // Step 3: Flatten normalized landmarks (63 values)
        val normalizedFlat = FloatArray(63)
        for (i in 0 until 21) {
            normalizedFlat[i * 3] = coords[i][0]
            normalizedFlat[i * 3 + 1] = coords[i][1]
            normalizedFlat[i * 3 + 2] = coords[i][2]
        }

        // Step 4: Distances (12 values)
        val distances = FloatArray(DISTANCE_PAIRS.size) { i ->
            distance(coords[DISTANCE_PAIRS[i].first], coords[DISTANCE_PAIRS[i].second])
        }

        // Step 5: Angles (9 values)
        val angles = FloatArray(ANGLE_TRIPLETS.size) { i ->
            angle(coords[ANGLE_TRIPLETS[i].first],
                coords[ANGLE_TRIPLETS[i].second],
                coords[ANGLE_TRIPLETS[i].third])
        }

        // Step 6: Extra features (6 values)
        val palmWidth = distance(coords[INDEX_MCP], coords[PINKY_MCP])
        val palmHeight = distance(coords[WRIST], coords[MIDDLE_MCP])
        val thumbIndexGap = distance(coords[THUMB_TIP], coords[INDEX_TIP])
        val indexPinkyGap = distance(coords[INDEX_TIP], coords[PINKY_TIP])
        val extraFeatures = floatArrayOf(
            palmWidth,
            palmHeight,
            thumbIndexGap,
            indexPinkyGap,
            palmWidth / (palmHeight + 1e-8f),
            thumbIndexGap / (palmWidth + 1e-8f)
        )

        // Step 7: Combine all — total 90 values
        return normalizedFlat + distances + angles + extraFeatures
    }

    private fun distance(a: FloatArray, b: FloatArray): Float {
        val dx = a[0] - b[0]
        val dy = a[1] - b[1]
        val dz = a[2] - b[2]
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    private fun angle(a: FloatArray, b: FloatArray, c: FloatArray): Float {
        val ba = floatArrayOf(a[0] - b[0], a[1] - b[1], a[2] - b[2])
        val bc = floatArrayOf(c[0] - b[0], c[1] - b[1], c[2] - b[2])
        val dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
        val denom = (norm(ba) + 1e-8f) * (norm(bc) + 1e-8f)
        val cosine = (dot / denom).coerceIn(-1f, 1f)
        return (acos(cosine) / Math.PI).toFloat()
    }

    private fun norm(v: FloatArray): Float {
        return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    }

    fun close() {
        interpreter?.close()
    }
}