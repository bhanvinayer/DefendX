package com.example.defend_x.utils

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt
import kotlin.random.Random

class TFLiteModel(private val context: Context) {
    private var interpreter: Interpreter? = null

    init {
        // try load model if exists; otherwise keep null and fallback to dummy
        try {
            val afd = context.assets.openFd("autoencoder.tflite")
            val inputStream = FileInputStream(afd.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = afd.startOffset
            val declaredLength = afd.declaredLength
            val mapped: MappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            interpreter = Interpreter(mapped)
        } catch (e: Exception) {
            e.printStackTrace()
            interpreter = null
        }
    }

    data class KeystrokeEvent(
        val key: Char,
        val pressTime: Long,
        val releaseTime: Long
    ) {
        val dwellTime: Long get() = releaseTime - pressTime
    }

    /**
     * Extract behavioral biometric features from typing pattern
     * Uses real keystroke timings, not simulated
     */
    private fun extractFeatures(typed: String, reference: String, keystrokeTimings: List<KeystrokeEvent>): FloatArray {
        if (keystrokeTimings.isEmpty()) {
            // Return default features if no timing data
            return floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f)
        }

        // Calculate dwell times (key press duration)
        val dwellTimes = keystrokeTimings.map { it.dwellTime.toFloat() }
        val avgDwellTime = dwellTimes.average().toFloat()
        val stdDwellTime = calculateStandardDeviation(dwellTimes)

        // Calculate flight times (time between key releases and next key presses)
        val flightTimes = mutableListOf<Float>()
        for (i in 0 until keystrokeTimings.size - 1) {
            val flightTime = keystrokeTimings[i + 1].pressTime - keystrokeTimings[i].releaseTime
            flightTimes.add(flightTime.toFloat())
        }
        val avgFlightTime = if (flightTimes.isNotEmpty()) flightTimes.average().toFloat() else 0f
        val stdFlightTime = if (flightTimes.isNotEmpty()) calculateStandardDeviation(flightTimes) else 0f

        // Count special characters
        val backspaceCount = typed.count { it == '\b' }.toFloat()
        val spaceCount = typed.count { it == ' ' }.toFloat()

        // Calculate keystroke rate (WPM - Words Per Minute)
        val totalTimeMs = if (keystrokeTimings.isNotEmpty()) {
            keystrokeTimings.last().releaseTime - keystrokeTimings.first().pressTime
        } else 1000L
        val totalTimeMinutes = totalTimeMs / 60000.0
        val wordCount = typed.split(" ").size.toFloat()
        val keystrokeRateWpm = if (totalTimeMinutes > 0) (wordCount / totalTimeMinutes).toFloat() else 0f

        return floatArrayOf(
            avgDwellTime,
            stdDwellTime,
            avgFlightTime,
            stdFlightTime,
            backspaceCount,
            spaceCount,
            keystrokeRateWpm
        )
    }

    private fun calculateStandardDeviation(values: List<Float>): Float {
        if (values.size <= 1) return 0f
        val mean = values.average()
        val variance = values.map { (it - mean) * (it - mean) }.average()
        return sqrt(variance).toFloat()
    }

    /**
     * Predict using autoencoder model with proper feature extraction
     *
     *        Use autoencoder -> Suspicious/Critical based on typing patterns
     *        Accepts real keystroke timings
     */
    fun predict(typed: String, reference: String, keystrokeTimings: List<KeystrokeEvent>): String {
        // Use real keystroke timings for feature extraction
        val features = extractFeatures(typed, reference, keystrokeTimings)

        interpreter?.let { interp ->
            try {
                // Prepare input tensor - autoencoder expects the 7 features
                val input = Array(1) { features }
                val output = Array(1) { FloatArray(7) }

                // Run inference
                interp.run(input, output)

                // Calculate reconstruction error
                val reconstructedFeatures = output[0]
                val reconstructionError = calculateReconstructionError(features, reconstructedFeatures)
                val confidence = String.format("%.2f", reconstructionError)

                // Detailed logging for debugging and threshold tuning
                println("[TFLiteModel] Reconstruction error (confidence): $confidence for input: '$typed'")
                println("[TFLiteModel] Features: ${features.joinToString()}")


                // Normal < 1.0, Suspicious < 1.5, Critical >= 1.5
                return when {
                    reconstructionError < 1.0 -> "Normal - Typing test successful (Confidence: $confidence)"
                    reconstructionError < 1.5 -> "Suspicious - Slight mismatch in typing (Confidence: $confidence)"
                    else -> {
                        val explanation = if (features[6] < 10f) {
                            "Critical - Unusual typing speed (Confidence: $confidence)"
                        } else {
                            "Critical - Unusual typing (Confidence: $confidence)"
                        }
                        explanation
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
                // Fallback if model fails
                return "Suspicious - Slight mismatch in typing (Confidence: N/A)"
            }
        }
        // Fallback if no model available
        return "Suspicious - Slight mismatch in typing (Confidence: N/A)"
    }

    private fun calculateReconstructionError(original: FloatArray, reconstructed: FloatArray): Float {
        if (original.size != reconstructed.size) return 1.0f

        var sumSquaredError = 0f
        for (i in original.indices) {
            val diff = original[i] - reconstructed[i]
            sumSquaredError += diff * diff
        }
        val rmse = sqrt(sumSquaredError / original.size)
        // Normalize RMSE to [0, 1] for confidence
        // Assume max reasonable RMSE is 1.5 (tune as needed)
        val normalized = (rmse / 1.5f).coerceIn(0f, 1f)
        return normalized
    }
}
