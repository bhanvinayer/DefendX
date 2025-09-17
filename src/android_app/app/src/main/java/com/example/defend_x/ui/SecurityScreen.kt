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
fun SecurityScreen(navController: NavController, context: Context) {
    var answer by remember { mutableStateOf("") }

    Scaffold(topBar = { TopAppBar(title = { Text("Set Security Answer") }) }) { padding ->
        Column(modifier = Modifier.fillMaxSize().padding(padding).padding(20.dp)) {
            Text("Enter your answer to security question:")
            OutlinedTextField(
                value = answer,
                onValueChange = { answer = it },
                modifier = Modifier.fillMaxWidth()
            )
            Button(
                onClick = {
                    val prefs = context.getSharedPreferences("defendx", Context.MODE_PRIVATE)
                    prefs.edit().putString("security_answer", answer).apply()
                    navController.popBackStack()
                },
                modifier = Modifier.padding(top = 16.dp)
            ) {
                Text("Save Answer")
            }
        }
    }
}
