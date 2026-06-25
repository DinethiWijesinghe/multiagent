#!/usr/bin/env python3
"""
Test suite for verifying data persistence across all backend features.
Run after api_server startup to validate PostgreSQL storage.

Usage:
    python test_persistence.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_USER_EMAIL = "persistence_test@example.com"
TEST_PASSWORD = "TestPass123!"
TEST_TOKEN = None

def log_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def log_test(name: str, status: str, details: str = ""):
    """Log test result."""
    icon = "✅" if status == "PASS" else "❌"
    print(f"{icon} {name}")
    if details:
        print(f"   {details}")

def test_user_registration() -> bool:
    """Test 1: User registration and persistence."""
    import requests
    
    log_section("TEST 1: User Registration & Profile Persistence")
    
    try:
        # Register new user
        resp = requests.post(
            f"{API_BASE_URL}/auth/register",
            json={
                "name": "Test Student",
                "email": TEST_USER_EMAIL,
                "password": TEST_PASSWORD,
            }
        )
        
        if resp.status_code != 200:
            log_test("Register user", "FAIL", f"Status {resp.status_code}: {resp.text}")
            return False
        
        log_test("Register user", "PASS", "User created successfully")
        
        # Login
        global TEST_TOKEN
        login_resp = requests.post(
            f"{API_BASE_URL}/auth/login",
            json={"email": TEST_USER_EMAIL, "password": TEST_PASSWORD}
        )
        
        if login_resp.status_code != 200:
            log_test("Login user", "FAIL", f"Status {login_resp.status_code}")
            return False
        
        data = login_resp.json()
        TEST_TOKEN = data.get("token")
        log_test("Login user", "PASS", f"Token obtained: {TEST_TOKEN[:20]}...")
        
        return True
    except Exception as e:
        log_test("User registration", "FAIL", str(e))
        return False


def test_profile_persistence() -> bool:
    """Test 2: User profile/state persistence."""
    import requests
    
    log_section("TEST 2: Profile State Persistence")
    
    if not TEST_TOKEN:
        log_test("Profile persistence", "FAIL", "No authentication token")
        return False
    
    try:
        headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
        
        # Save profile state
        profile_data = {
            "user_id": TEST_USER_EMAIL,
            "state": {
                "first_name": "Test",
                "last_name": "Student",
                "target_country": "UK",
                "annual_budget": 50000,
                "education_level": "High School",
                "academic_background": "A-Levels",
            }
        }
        
        resp = requests.post(
            f"{API_BASE_URL}/user/state",
            json=profile_data,
            headers=headers
        )
        
        if resp.status_code != 200:
            log_test("Save profile state", "FAIL", f"Status {resp.status_code}")
            return False
        
        log_test("Save profile state", "PASS", "Profile saved to PostgreSQL")
        
        # Retrieve profile state
        resp = requests.get(
            f"{API_BASE_URL}/user/state?user_id={TEST_USER_EMAIL}",
            headers=headers
        )
        
        if resp.status_code != 200:
            log_test("Load profile state", "FAIL", f"Status {resp.status_code}")
            return False
        
        loaded = resp.json().get("state", {})
        
        # Verify all fields persisted
        checks = [
            ("First name", loaded.get("first_name") == "Test"),
            ("Last name", loaded.get("last_name") == "Student"),
            ("Target country", loaded.get("target_country") == "UK"),
            ("Annual budget", loaded.get("annual_budget") == 50000),
        ]
        
        all_pass = all(check[1] for check in checks)
        for field, passed in checks:
            log_test(f"  Verify {field}", "PASS" if passed else "FAIL")
        
        return all_pass
        
    except Exception as e:
        log_test("Profile persistence", "FAIL", str(e))
        return False


def test_chat_history_persistence() -> bool:
    """Test 3: Chat history persistence."""
    import requests
    
    log_section("TEST 3: Chat History Persistence")
    
    if not TEST_TOKEN:
        log_test("Chat history", "FAIL", "No authentication token")
        return False
    
    try:
        headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
        
        # Send chat message
        messages = [
            {
                "id": f"msg_{int(time.time())}",
                "role": "user",
                "text": "What universities can I apply to with my GPA?",
                "time": time.time(),
            }
        ]
        
        resp = requests.post(
            f"{API_BASE_URL}/chat/history",
            json={
                "user_id": TEST_USER_EMAIL,
                "messages": messages,
                "agent_data": {"test": "data"},
            },
            headers=headers
        )
        
        if resp.status_code != 200:
            log_test("Save chat message", "FAIL", f"Status {resp.status_code}")
            return False
        
        log_test("Save chat message", "PASS", "Message saved to PostgreSQL")
        
        # Retrieve chat history
        resp = requests.get(
            f"{API_BASE_URL}/chat/history?user_id={TEST_USER_EMAIL}",
            headers=headers
        )
        
        if resp.status_code != 200:
            log_test("Load chat history", "FAIL", f"Status {resp.status_code}")
            return False
        
        data = resp.json()
        loaded_messages = data.get("messages", [])
        
        if len(loaded_messages) == 0:
            log_test("Load chat history", "FAIL", "No messages in response")
            return False
        
        log_test("Load chat history", "PASS", f"Retrieved {len(loaded_messages)} messages")
        return True
        
    except Exception as e:
        log_test("Chat history", "FAIL", str(e))
        return False


def test_document_upload() -> bool:
    """Test 4: Document upload persistence."""
    import requests
    
    log_section("TEST 4: Document Upload Persistence")
    
    if not TEST_TOKEN:
        log_test("Document upload", "FAIL", "No authentication token")
        return False
    
    try:
        headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
        
        # Create a test image file
        test_image_path = Path("/tmp/test_doc.jpg")
        
        # Create a minimal JPEG (1x1 white pixel)
        test_image_path.write_bytes(
            b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
            b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c'
            b'\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c'
            b'\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00'
            b'\x01\x00\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01'
            b'\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06'
            b'\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd7\xff\xd9'
        )
        
        # Upload document
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            resp = requests.post(
                f"{API_BASE_URL}/documents/upload",
                files=files,
                headers=headers
            )
        
        if resp.status_code != 200:
            log_test("Upload document", "FAIL", f"Status {resp.status_code}")
            return False
        
        data = resp.json()
        document_id = data.get("document", {}).get("document_id")
        log_test("Upload document", "PASS", f"Document saved: {document_id}")
        
        # List documents
        resp = requests.get(
            f"{API_BASE_URL}/documents",
            headers=headers
        )
        
        if resp.status_code != 200:
            log_test("List documents", "FAIL", f"Status {resp.status_code}")
            return False
        
        docs = resp.json().get("documents", [])
        if len(docs) == 0:
            log_test("List documents", "FAIL", "No documents returned")
            return False
        
        log_test("List documents", "PASS", f"Retrieved {len(docs)} documents from PostgreSQL")
        
        # Clean up
        test_image_path.unlink()
        return True
        
    except Exception as e:
        log_test("Document upload", "FAIL", str(e))
        return False


def test_application_submission() -> bool:
    """Test 5: Application submission persistence."""
    import requests
    
    log_section("TEST 5: Application Submission Persistence")
    
    if not TEST_TOKEN:
        log_test("Application submission", "FAIL", "No authentication token")
        return False
    
    try:
        headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
        
        # Submit application
        resp = requests.post(
            f"{API_BASE_URL}/applications",
            json={
                "university_name": "University of Cambridge",
                "university_id": "uk_cambridge",
                "program": "Computer Science",
                "country": "UK",
                "eligibility_tier": "direct",
                "grade_point": 3.8,
            },
            headers=headers
        )
        
        if resp.status_code != 200:
            log_test("Submit application", "FAIL", f"Status {resp.status_code}")
            return False
        
        app_data = resp.json().get("application", {})
        app_id = app_data.get("application_id")
        log_test("Submit application", "PASS", f"Application saved: {app_id}")
        
        # List applications
        resp = requests.get(
            f"{API_BASE_URL}/applications",
            headers=headers
        )
        
        if resp.status_code != 200:
            log_test("List applications", "FAIL", f"Status {resp.status_code}")
            return False
        
        apps = resp.json().get("applications", [])
        if len(apps) == 0:
            log_test("List applications", "FAIL", "No applications returned")
            return False
        
        log_test("List applications", "PASS", f"Retrieved {len(apps)} applications from PostgreSQL")
        
        # Get specific application
        resp = requests.get(
            f"{API_BASE_URL}/applications/{app_id}",
            headers=headers
        )
        
        if resp.status_code != 200:
            log_test("Get application details", "FAIL", f"Status {resp.status_code}")
            return False
        
        log_test("Get application details", "PASS", "Application details retrieved")
        
        return True
        
    except Exception as e:
        log_test("Application submission", "FAIL", str(e))
        return False


def test_pdf_support() -> bool:
    """Test 6: PDF file upload support."""
    import requests
    
    log_section("TEST 6: PDF File Upload Support")
    
    if not TEST_TOKEN:
        log_test("PDF support", "FAIL", "No authentication token")
        return False
    
    try:
        headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
        
        # Check if pdf2image is available
        try:
            from pdf2image import convert_from_path
            log_test("pdf2image dependency", "PASS", "Module available")
        except ImportError:
            log_test("pdf2image dependency", "FAIL", "Run: pip install pdf2image")
            return False
        
        # Create a minimal PDF for testing
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            pdf_path = Path("/tmp/test_document.pdf")
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            c.drawString(100, 750, "UNIVERSITY OF CAMBRIDGE")
            c.drawString(100, 720, "Bachelor of Science - Computer Science")
            c.drawString(100, 690, "Grade: First Class Honours")
            c.save()
            
            log_test("Create test PDF", "PASS", "PDF generated")
            
            # Upload PDF to OCR endpoint
            with open(pdf_path, 'rb') as f:
                files = {'file': ('test.pdf', f, 'application/pdf')}
                resp = requests.post(
                    f"{API_BASE_URL}/ocr",
                    files=files,
                    headers=headers
                )
            
            if resp.status_code == 200:
                log_test("Upload PDF to OCR", "PASS", "PDF processed successfully")
                pdf_path.unlink()
                return True
            elif resp.status_code == 503:
                log_test("Upload PDF to OCR", "FAIL", "OCR engine not available")
                pdf_path.unlink()
                return False
            else:
                log_test("Upload PDF to OCR", "FAIL", f"Status {resp.status_code}")
                pdf_path.unlink()
                return False
                
        except ImportError:
            log_test("reportlab dependency", "FAIL", "Run: pip install reportlab")
            return False
        
    except Exception as e:
        log_test("PDF support", "FAIL", str(e))
        return False


def main():
    """Run all persistence tests."""
    import requests
    
    log_section("UniAssist Backend - Data Persistence Test Suite")
    
    # Verify API is reachable
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if resp.status_code != 200:
            print("❌ API server is not responding. Start with:")
            print("   uvicorn multiagent.api_server:app --host 0.0.0.0 --port 8000")
            sys.exit(1)
        log_test("API server connectivity", "PASS", "Server responding on port 8000")
    except Exception as e:
        print(f"❌ Cannot connect to API server: {e}")
        print("   Start with: uvicorn multiagent.api_server:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    # Run tests
    results = []
    results.append(("User Registration", test_user_registration()))
    results.append(("Profile Persistence", test_profile_persistence()))
    results.append(("Chat History", test_chat_history_persistence()))
    results.append(("Document Upload", test_document_upload()))
    results.append(("Application Submission", test_application_submission()))
    results.append(("PDF Support", test_pdf_support()))
    
    # Summary
    log_section("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All persistence features are working correctly!")
        print("\nSupported file types for OCR: .pdf, .jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
