package com.example.defend_x.ui.screens

import android.content.Context
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VerifyScreen(navController: NavController, context: Context) {
    val prefs = context.getSharedPreferences("defendx", Context.MODE_PRIVATE)
    val savedAnswer = prefs.getString("security_answer", "") ?: ""

    var input by remember { mutableStateOf("") }
    var message by remember { mutableStateOf("") }

    Scaffold(topBar = { TopAppBar(title = { Text("Verify Identity") }) }) { padding ->
        Column(modifier = Modifier.fillMaxSize().padding(padding).padding(20.dp)) {
            Text("Answer your security question:")
            OutlinedTextField(value = input, onValueChange = { input = it }, modifier = Modifier.fillMaxWidth())

            Button(onClick = {
                if (input == savedAnswer) {
                    message = "Verified ✅"
                    navController.navigate("home") {
                        popUpTo("home") { inclusive = true }
                    }
                } else {
                    message = "Wrong Answer ❌"
                }
            }, modifier = Modifier.padding(top = 16.dp)) {
                Text("Verify")
            }

            if (message.isNotEmpty()) {
                Text(message, modifier = Modifier.padding(top = 20.dp))
            }
        }
    }
}
