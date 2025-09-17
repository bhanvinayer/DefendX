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
     */
    private fun extractFeatures(typed: String, reference: String, simulatedTimings: List<KeystrokeEvent>): FloatArray {
        if (simulatedTimings.isEmpty()) {
            // Return default features if no timing data
            return floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f)
        }

        // Calculate dwell times (key press duration)
        val dwellTimes = simulatedTimings.map { it.dwellTime.toFloat() }
        val avgDwellTime = dwellTimes.average().toFloat()
        val stdDwellTime = calculateStandardDeviation(dwellTimes)

        // Calculate flight times (time between key releases and next key presses)
        val flightTimes = mutableListOf<Float>()
        for (i in 0 until simulatedTimings.size - 1) {
            val flightTime = simulatedTimings[i + 1].pressTime - simulatedTimings[i].releaseTime
            flightTimes.add(flightTime.toFloat())
        }

        val avgFlightTime = if (flightTimes.isNotEmpty()) flightTimes.average().toFloat() else 0f
        val stdFlightTime = if (flightTimes.isNotEmpty()) calculateStandardDeviation(flightTimes) else 0f

        // Count special characters
        val backspaceCount = typed.count { it == '\b' }.toFloat()
        val spaceCount = typed.count { it == ' ' }.toFloat()

        // Calculate keystroke rate (WPM - Words Per Minute)
        val totalTimeMs = if (simulatedTimings.isNotEmpty()) {
            simulatedTimings.last().releaseTime - simulatedTimings.first().pressTime
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
     * Simulate keystroke timings based on typing characteristics
     * In a real implementation, you would capture actual keystroke events
     */
    private fun simulateKeystrokeTimings(text: String): List<KeystrokeEvent> {
        val events = mutableListOf<KeystrokeEvent>()
        var currentTime = System.currentTimeMillis()

        text.forEach { char ->
            // Simulate realistic typing patterns
            val baseDwellTime = when {
                char.isLetter() -> Random.nextLong(80, 150)
                char.isDigit() -> Random.nextLong(100, 180)
                char == ' ' -> Random.nextLong(60, 120)
                else -> Random.nextLong(90, 160)
            }

            val baseFlightTime = Random.nextLong(50, 200)

            val pressTime = currentTime
            val releaseTime = pressTime + baseDwellTime

            events.add(KeystrokeEvent(char, pressTime, releaseTime))
            currentTime = releaseTime + baseFlightTime
        }

        return events
    }

    /**
     * Predict using autoencoder model with proper feature extraction
     *
     *        Use autoencoder -> Suspicious/Critical based on typing patterns
     */
    fun predict(typed: String, reference: String): String {
        // sentence entered
        if (typed.trim().equals(reference.trim(), ignoreCase = true)) {
            return "Normal - Typing test successful"
        }

        //  use autoencoder to analyze typing behavior
        val keystrokeTimings = simulateKeystrokeTimings(typed)
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

                // Classification based on autoencoder reconstruction error
                return when {
                    reconstructionError < 0.5 -> "Suspicious - Slight mismatch in typing"
                    else -> "Critical - Unusual typing"
                }

            } catch (e: Exception) {
                e.printStackTrace()
                // Fallback if model fails
                return "Suspicious - Slight mismatch in typing"
            }
        }

        // Fallback if no model available
        return "Suspicious - Slight mismatch in typing"
    }

    private fun calculateReconstructionError(original: FloatArray, reconstructed: FloatArray): Float {
        if (original.size != reconstructed.size) return 1.0f

        var sumSquaredError = 0f
        for (i in original.indices) {
            val diff = original[i] - reconstructed[i]
            sumSquaredError += diff * diff
        }

        return sqrt(sumSquaredError / original.size)
    }
}
