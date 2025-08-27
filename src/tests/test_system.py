"""
Test script for the fraud detection system
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import *
from utils import *
from advanced_models import *
from config import *

class TestKeystrokeAgent(unittest.TestCase):
    
    def setUp(self):
        self.agent = KeystrokeAgent()
    
    def test_keystroke_capture(self):
        """Test keystroke capture functionality"""
        self.agent.start_capture()
        
        # Simulate typing
        test_text = "hello"
        for i, char in enumerate(test_text):
            self.agent.process_keystroke(char, time.time() + i * 0.1)
        
        self.assertEqual(len(self.agent.key_events), len(test_text))
        self.assertIsNotNone(self.agent.start_time)
    
    def test_feature_extraction(self):
        """Test feature extraction from keystroke data"""
        self.agent.start_capture()
        
        test_text = "The quick brown fox"
        reference_text = "The quick brown fox"
        
        for i, char in enumerate(test_text):
            self.agent.process_keystroke(char, time.time() + i * 0.1)
        
        features = self.agent.extract_features(test_text, reference_text)
        
        # Check that all required features are present
        required_features = ['wpm', 'avg_flight_time', 'accuracy', 'total_time']
        for feature in required_features:
            self.assertIn(feature, features)
        
        # Check accuracy calculation
        self.assertEqual(features['accuracy'], 100.0)

class TestBehaviorModelAgent(unittest.TestCase):
    
    def setUp(self):
        self.agent = BehaviorModelAgent()
    
    def test_user_profile_creation(self):
        """Test user profile creation"""
        user_id = "test_user"
        features = {
            'wpm': 50,
            'accuracy': 95,
            'avg_flight_time': 0.2,
            'std_flight_time': 0.05
        }
        
        self.agent.create_user_profile(user_id, features)
        
        self.assertIn(user_id, self.agent.user_profiles)
        self.assertEqual(len(self.agent.user_profiles[user_id]['baseline_features']), 1)
    
    def test_model_training(self):
        """Test model training with sufficient samples"""
        user_id = "test_user"
        
        # Add multiple baseline samples
        for i in range(5):
            features = {
                'wpm': 50 + np.random.uniform(-5, 5),
                'accuracy': 95 + np.random.uniform(-2, 2),
                'avg_flight_time': 0.2 + np.random.uniform(-0.02, 0.02),
                'std_flight_time': 0.05 + np.random.uniform(-0.01, 0.01),
                'typing_rhythm_variance': 0.03 + np.random.uniform(-0.005, 0.005),
                'max_flight_time': 0.3 + np.random.uniform(-0.05, 0.05),
                'min_flight_time': 0.1 + np.random.uniform(-0.02, 0.02)
            }
            self.agent.create_user_profile(user_id, features)
        
        # Train model
        success = self.agent.train_user_model(user_id)
        
        self.assertTrue(success)
        self.assertIn(user_id, self.agent.user_models)
        self.assertTrue(self.agent.user_profiles[user_id]['model_trained'])

class TestFraudDetectionAgent(unittest.TestCase):
    
    def setUp(self):
        self.agent = FraudDetectionAgent()
    
    def test_fraud_analysis(self):
        """Test fraud analysis functionality"""
        user_id = "test_user"
        features = {
            'wpm': 150,  # Unusually high
            'accuracy': 60,  # Low accuracy
            'avg_flight_time': 0.1,
            'typing_rhythm_variance': 0.8  # High variance
        }
        
        analysis = self.agent.analyze_session(user_id, features, True, 0.8)
        
        self.assertIn('fraud_detected', analysis)
        self.assertIn('risk_score', analysis)
        self.assertIn('risk_factors', analysis)
        
        # Should detect fraud due to high speed and low accuracy
        self.assertTrue(analysis['fraud_detected'])
        self.assertGreater(analysis['risk_score'], self.agent.fraud_threshold)

class TestDataManagerAgent(unittest.TestCase):
    
    def setUp(self):
        self.agent = DataManagerAgent("test_data")
    
    def tearDown(self):
        # Clean up test data
        import shutil
        if os.path.exists("test_data"):
            shutil.rmtree("test_data")
    
    def test_user_data_persistence(self):
        """Test saving and loading user data"""
        user_id = "test_user"
        test_data = {
            'baseline_features': [{'wpm': 50, 'accuracy': 95}],
            'model_trained': True
        }
        
        # Save data
        self.agent.save_user_data(user_id, test_data)
        
        # Load data
        loaded_data = self.agent.load_user_data(user_id)
        
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data['model_trained'], True)
        self.assertEqual(len(loaded_data['baseline_features']), 1)

class TestUtils(unittest.TestCase):
    
    def test_feature_extractor(self):
        """Test advanced feature extraction"""
        key_events = [
            {'char': 'h', 'flight_time': 0.1},
            {'char': 'e', 'flight_time': 0.15},
            {'char': 'l', 'flight_time': 0.12},
            {'char': 'l', 'flight_time': 0.11},
            {'char': 'o', 'flight_time': 0.13}
        ]
        
        ngram_features = FeatureExtractor.extract_ngram_features(key_events, 2)
        
        self.assertIn('ngram_2_mean', ngram_features)
        self.assertIn('ngram_2_std', ngram_features)
    
    def test_data_validation(self):
        """Test data validation"""
        valid_features = {
            'wpm': 50,
            'accuracy': 95,
            'avg_flight_time': 0.2
        }
        
        is_valid, errors = DataValidation.validate_typing_session(valid_features)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Test invalid features
        invalid_features = {
            'wpm': 500,  # Too high
            'accuracy': 150,  # Over 100%
            'avg_flight_time': -0.1  # Negative
        }
        
        is_valid, errors = DataValidation.validate_typing_session(invalid_features)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

class TestAdvancedModels(unittest.TestCase):
    
    def test_advanced_anomaly_detector(self):
        """Test advanced anomaly detection"""
        # Generate sample training data
        np.random.seed(42)
        X_train = np.random.normal(0, 1, (50, 7))  # 50 samples, 7 features
        feature_names = ['wpm', 'accuracy', 'avg_flight_time', 'std_flight_time', 
                        'typing_rhythm_variance', 'max_flight_time', 'min_flight_time']
        
        detector = AdvancedAnomalyDetector()
        detector.fit(X_train, feature_names)
        
        # Test prediction
        X_test = np.random.normal(0, 1, (10, 7))
        predictions, scores = detector.predict(X_test)
        
        self.assertEqual(len(predictions), 10)
        self.assertEqual(len(scores), 10)
        self.assertIsInstance(predictions[0], bool)
    
    def test_behavior_evolution_tracker(self):
        """Test behavior evolution tracking"""
        tracker = BehaviorEvolutionTracker(window_size=5)
        user_id = "test_user"
        
        # Add some historical behavior
        for i in range(3):
            features = {'wpm': 50 + i, 'accuracy': 95 - i}
            tracker.update_user_behavior(user_id, features)
        
        # Test drift detection
        new_features = {'wpm': 80, 'accuracy': 85}  # Significant change
        drift_detected, magnitude = tracker.detect_behavior_drift(user_id, new_features)
        
        self.assertIsInstance(drift_detected, bool)
        self.assertIsInstance(magnitude, float)
        self.assertGreaterEqual(magnitude, 0.0)
        self.assertLessEqual(magnitude, 1.0)

def run_system_tests():
    """Run comprehensive system tests"""
    print("🧪 Running On-Device Fraud Detection System Tests")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestKeystrokeAgent,
        TestBehaviorModelAgent,
        TestFraudDetectionAgent,
        TestDataManagerAgent,
        TestUtils,
        TestAdvancedModels
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n⚠️ ERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if not result.failures and not result.errors:
        print("\n✅ All tests passed successfully!")
    
    return result.wasSuccessful()

def test_integration():
    """Test full system integration"""
    print("\n🔄 Running Integration Tests")
    print("-" * 30)
    
    try:
        # Test agent initialization
        keystroke_agent = KeystrokeAgent()
        behavior_agent = BehaviorModelAgent()
        fraud_agent = FraudDetectionAgent()
        data_agent = DataManagerAgent("test_integration")
        
        print("✅ Agent initialization successful")
        
        # Test user registration flow
        user_id = "integration_test_user"
        
        # Simulate baseline collection
        for i in range(3):
            keystroke_agent.start_capture()
            
            # Simulate typing
            test_text = "The quick brown fox jumps over the lazy dog."
            for j, char in enumerate(test_text):
                keystroke_agent.process_keystroke(char, time.time() + j * 0.1)
            
            features = keystroke_agent.extract_features(test_text, test_text)
            behavior_agent.create_user_profile(user_id, features)
        
        print("✅ User registration simulation successful")
        
        # Test model training
        success = behavior_agent.train_user_model(user_id)
        if success:
            print("✅ Model training successful")
        else:
            print("❌ Model training failed")
            return False
        
        # Test verification flow
        keystroke_agent.start_capture()
        test_text = "Authentication test sentence."
        
        for i, char in enumerate(test_text):
            keystroke_agent.process_keystroke(char, time.time() + i * 0.1)
        
        features = keystroke_agent.extract_features(test_text, test_text)
        is_anomaly, confidence = behavior_agent.predict_anomaly(user_id, features)
        
        analysis = fraud_agent.analyze_session(user_id, features, is_anomaly, confidence)
        
        print("✅ Verification flow successful")
        print(f"   - Anomaly detected: {is_anomaly}")
        print(f"   - Confidence: {confidence:.3f}")
        print(f"   - Fraud detected: {analysis['fraud_detected']}")
        print(f"   - Risk score: {analysis['risk_score']:.3f}")
        
        # Test data persistence
        profile_data = behavior_agent.user_profiles[user_id]
        data_agent.save_user_data(user_id, profile_data)
        
        loaded_data = data_agent.load_user_data(user_id)
        if loaded_data:
            print("✅ Data persistence successful")
        else:
            print("❌ Data persistence failed")
            return False
        
        # Cleanup
        import shutil
        if os.path.exists("test_integration"):
            shutil.rmtree("test_integration")
        
        print("✅ Integration tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Fraud Detection System Test Suite")
    print("=" * 60)
    
    # Run unit tests
    unit_tests_passed = run_system_tests()
    
    # Run integration tests
    integration_tests_passed = test_integration()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Unit Tests: {'✅ PASSED' if unit_tests_passed else '❌ FAILED'}")
    print(f"Integration Tests: {'✅ PASSED' if integration_tests_passed else '❌ FAILED'}")
    
    if unit_tests_passed and integration_tests_passed:
        print("\n🎉 ALL TESTS PASSED - System is ready for use!")
        print("\nTo start the application, run:")
        print("streamlit run app.py")
    else:
        print("\n⚠️ Some tests failed - Please review before using the system")
    
    print("=" * 60)
