package com.example.defend_x.ui.screens

import android.content.Context
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.key.*
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.example.defend_x.utils.TFLiteModel
import com.example.defend_x.utils.TFLiteModel.KeystrokeEvent
import com.example.defend_x.utils.NotificationHelper

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TestScreen(navController: NavController, context: Context) {
    var input by remember { mutableStateOf("") }
    var result by remember { mutableStateOf("") }
    var showSecurityQuestion by remember { mutableStateOf(false) }
    var securityAnswer by remember { mutableStateOf("") }
    var finalResult by remember { mutableStateOf("") }
    val testSentence = "the quick brown fox jumps over the lazy dog"

    // List to store real keystroke events
    val keystrokeEvents = remember { mutableStateListOf<KeystrokeEvent>() }
    // Map to track press times for each key
    val keyPressTimes = remember { mutableStateMapOf<Char, Long>() }

    Scaffold(topBar = { TopAppBar(title = { Text("Behavioral Biometric Authentication") }) }) { padding ->
        Column(modifier = Modifier.fillMaxSize().padding(padding).padding(20.dp)) {
            Text("Type this sentence:")
            Text(testSentence, modifier = Modifier.padding(top = 8.dp))

            OutlinedTextField(
                value = input,
                onValueChange = { input = it },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 10.dp)
                    .onKeyEvent { event ->
                        val keyChar = event.key.nativeKeyCode.toChar()
                        val now = System.currentTimeMillis()
                        when (event.type) {
                            KeyEventType.KeyDown -> {
                                keyPressTimes[keyChar] = now
                            }
                            KeyEventType.KeyUp -> {
                                val pressTime = keyPressTimes[keyChar] ?: now
                                keystrokeEvents.add(KeystrokeEvent(keyChar, pressTime, now))
                                keyPressTimes.remove(keyChar)
                            }
                        }
                        false
                    }
            )

            Button(onClick = {
                val model = TFLiteModel(context)
                val pred = model.predict(input, testSentence, keystrokeEvents.toList())
                result = pred

                // Check if suspicious or critical - show security question
                if (pred.startsWith("Suspicious") || pred.startsWith("Critical")) {
                    showSecurityQuestion = true
                    finalResult = ""
                } else {
                    // Normal result - no security question needed
                    finalResult = pred
                    showSecurityQuestion = false
                }
            }, modifier = Modifier.padding(top = 16.dp)) {
                Text("Submit")
            }

            // Show security question if suspicious/critical
            if (showSecurityQuestion) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 20.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            "Security Verification Required",
                            style = MaterialTheme.typography.titleMedium,
                            color = MaterialTheme.colorScheme.onErrorContainer
                        )
                        Text(
                            "Initial Result: $result",
                            modifier = Modifier.padding(top = 8.dp),
                            color = MaterialTheme.colorScheme.onErrorContainer
                        )
                        Text(
                            "Please answer your security question:",
                            modifier = Modifier.padding(top = 12.dp),
                            color = MaterialTheme.colorScheme.onErrorContainer
                        )

                        OutlinedTextField(
                            value = securityAnswer,
                            onValueChange = { securityAnswer = it },
                            label = { Text("Security Answer") },
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 8.dp)
                        )

                        Button(
                            onClick = {
                                val prefs = context.getSharedPreferences("defendx", Context.MODE_PRIVATE)
                                val savedAnswer = prefs.getString("security_answer", "") ?: ""

                                if (securityAnswer.trim().equals(savedAnswer.trim(), ignoreCase = true)) {
                                    finalResult = "Verified - Security question answered correctly"
                                } else {
                                    finalResult = "Locked out - Incorrect security answer"
                                }
                                showSecurityQuestion = false
                                securityAnswer = ""
                            },
                            modifier = Modifier.padding(top = 12.dp)
                        ) {
                            Text("Verify")
                        }
                    }
                }
            }

            // Show final result
            if (finalResult.isNotEmpty()) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 20.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = when {
                            finalResult.startsWith("Normal") || finalResult.startsWith("Verified") ->
                                MaterialTheme.colorScheme.primaryContainer
                            finalResult.startsWith("Locked out") ->
                                MaterialTheme.colorScheme.errorContainer
                            else -> MaterialTheme.colorScheme.secondaryContainer
                        }
                    )
                ) {
                    Text(
                        "Result: $finalResult",
                        modifier = Modifier.padding(16.dp),
                        style = MaterialTheme.typography.titleMedium
                    )
                }
            }

            // Show sample workspace table only if result is Normal or Verified
            if (finalResult.startsWith("Normal") || finalResult.startsWith("Verified")) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 20.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Sample Workspace Database", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(Modifier.fillMaxWidth()) {
                            Text("User", modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
                            Text("Status", modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
                            Text("Last Login", modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
                        }
                        Divider()
                        Row(Modifier.fillMaxWidth()) {
                            Text("alice", modifier = Modifier.weight(1f))
                            Text("Active", modifier = Modifier.weight(1f))
                            Text("2025-09-15", modifier = Modifier.weight(1f))
                        }
                        Row(Modifier.fillMaxWidth()) {
                            Text("bob", modifier = Modifier.weight(1f))
                            Text("Inactive", modifier = Modifier.weight(1f))
                            Text("2025-09-10", modifier = Modifier.weight(1f))
                        }
                        Row(Modifier.fillMaxWidth()) {
                            Text("carol", modifier = Modifier.weight(1f))
                            Text("Active", modifier = Modifier.weight(1f))
                            Text("2025-09-16", modifier = Modifier.weight(1f))
                        }
                    }
                }
            }
        }
    }
}
