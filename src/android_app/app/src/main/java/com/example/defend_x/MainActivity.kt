package com.example.defend_x

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.defend_x.ui.screens.*
import com.example.defend_x.ui.theme.DEFEND_XTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            DEFEND_XTheme {
                val navController = rememberNavController()

                // Handle intent if launched from notification
                LaunchedEffect(key1 = intent) {
                    if (intent?.getBooleanExtra("open_verify", false) == true) {
                        navController.navigate("verify")
                    }
                }

                NavHost(navController = navController, startDestination = "home") {
                    composable("home") { HomeScreen(navController) }
                    composable("security") { SecurityScreen(navController, this@MainActivity) }
                    composable("test") { TestScreen(navController, this@MainActivity) }
                    composable("verify") { VerifyScreen(navController, this@MainActivity) }
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
    }
}
