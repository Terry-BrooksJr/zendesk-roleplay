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
        # Valid payload example
        payload = {"user_id": "123", "session_data": {"foo": "bar"}}
        response = requests.post("http://localhost:8000/sessions", json=payload)
        assert response.status_code == 201, f"Expected 201, got {response.status_code}"
        print("✓ Session creation with valid payload passed")

        # Missing required field: user_id
        payload_missing_user_id = {"session_data": {"foo": "bar"}}
        response = requests.post(
            "http://localhost:8000/sessions", json=payload_missing_user_id
        )
        assert (
            response.status_code == 400
        ), f"Expected 400 for missing user_id, got {response.status_code}"
        assert (
            "user_id" in response.text
        ), "Error message should mention missing user_id"
        print("✓ Session creation with missing user_id correctly failed")

        # Missing required field: session_data
        payload_missing_session_data = {"user_id": "123"}
        response = requests.post(
            "http://localhost:8000/sessions", json=payload_missing_session_data
        )
        assert (
            response.status_code == 400
        ), f"Expected 400 for missing session_data, got {response.status_code}"
        assert (
            "session_data" in response.text
        ), "Error message should mention missing session_data"
        print("✓ Session creation with missing session_data correctly failed")

        # Invalid payload: user_id is not a string
        payload_invalid_user_id = {
            "user_id": 123,  # should be string
            "session_data": {"foo": "bar"},
        }
        response = requests.post(
            "http://localhost:8000/sessions", json=payload_invalid_user_id
        )
        assert (
            response.status_code == 400
        ), f"Expected 400 for invalid user_id type, got {response.status_code}"
        assert (
            "user_id" in response.text
        ), "Error message should mention invalid user_id type"
        print("✓ Session creation with invalid user_id type correctly failed")

        # Invalid payload: session_data is not a dict
        payload_invalid_session_data = {"user_id": "123", "session_data": "not_a_dict"}
        response = requests.post(
            "http://localhost:8000/sessions", json=payload_invalid_session_data
        )
        assert (
            response.status_code == 400
        ), f"Expected 400 for invalid session_data type, got {response.status_code}"
        assert (
            "session_data" in response.text
        ), "Error message should mention invalid session_data type"
        print("✓ Session creation with invalid session_data type correctly failed")

    except Exception as e:
        print(f"✗ Session creation test failed: {e}")
        assert False
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
        # Normal message
        payload = {"session_id": session_id, "message": "Hello, world!"}
        response = requests.post("http://localhost:8000/chat", json=payload)
        if response.status_code == 200:
            print("✓ Chat message sent successfully")
        else:
            print(f"✗ Chat message failed: {response.status_code}")
            print(f"Response: {response.text}")

        # Edge case: Empty message
        payload_empty = {"session_id": session_id, "message": ""}
        response_empty = requests.post("http://localhost:8000/chat", json=payload_empty)
        if response_empty.status_code != 200:
            print("✓ Empty message correctly rejected")
        else:
            print("✗ Empty message was accepted (should be rejected)")

        # Edge case: Invalid session ID
        payload_invalid = {"session_id": "invalid_session_id", "message": "Test"}
        response_invalid = requests.post(
            "http://localhost:8000/chat", json=payload_invalid
        )
        if response_invalid.status_code != 200:
            print("✓ Invalid session ID correctly rejected")
        else:
            print("✗ Invalid session ID was accepted (should be rejected)")

        # Edge case: Rate limiting
        rate_limit_triggered = False
        for i in range(10):  # Adjust number as needed for your rate limit
            payload_rate = {"session_id": session_id, "message": f"Spam message {i}"}
            response_rate = requests.post(
                "http://localhost:8000/chat", json=payload_rate
            )
            if response_rate.status_code == 429:
                print("✓ Rate limiting triggered as expected")
                rate_limit_triggered = True
                break
        if not rate_limit_triggered:
            print("✗ Rate limiting not triggered (check rate limit config)")

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
