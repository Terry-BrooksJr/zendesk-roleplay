#!/usr/bin/env python3
"""
Quick test script to verify the application is working correctly.
"""

import requests


def test_health():
    """Test the health endpoint."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✓ Health endpoint working")
            return True
        else:
            print(f"✗ Health endpoint returned {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to application (is it running on port 8000?)")
        return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_session_creation():
    """Test creating a new session."""
    try:
        response = requests.post(
            "http://localhost:8000/start",
            json={"candidate_label": "test_user"},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            session_id = data.get("session_id")
            if session_id:
                print(f"✓ Session created: {session_id}")
                return session_id
            else:
                print("✗ No session_id in response")
                return None
        else:
            print(f"✗ Session creation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Session creation failed: {e}")
        return None


def test_chat_message(session_id):
    """Test sending a chat message."""
    try:
        response = requests.post(
            "http://localhost:8000/reply",
            json={
                "session_id": session_id,
                "message": "Hello, I need help with testing",
            },
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            message = data.get("message", "")
            if message:
                print(f"✓ Chat response received: {message[:100]}...")
                return True
            else:
                print("✗ No message in chat response")
                return False
        else:
            print(f"✗ Chat failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Chat failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing chatbot application...")
    print("=" * 50)

    # Test 1: Health check
    if not test_health():
        print("\nApplication is not running. Start it with: python app.py")
        return

    # Test 2: Session creation
    session_id = test_session_creation()
    if not session_id:
        print("\nSession creation failed. Check logs for errors.")
        return

    # Test 3: Chat functionality
    if test_chat_message(session_id):
        print("\n🎉 All tests passed! Application is working correctly.")
    else:
        print("\nChat functionality failed. Check logs for errors.")


if __name__ == "__main__":
    main()
