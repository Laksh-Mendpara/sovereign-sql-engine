import requests
import json
import argparse

def test_inference(stream=False):
    url = "http://localhost:8000/runsync"
    
    # RunPod worker local testing payload
    payload = {
        "input": {
            "prompt": "Hello! Please explain what a SQL engine does in simple terms.",
            "stream": stream,
            "sampling_params": {
                "temperature": 0.7,
                "max_tokens": 150
            }
        }
    }

    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload, stream=stream)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return

        if stream:
            print("Streaming output:")
            # RunPod local testing stream is an array of chunks or newline separated.
            # Local test endpoints might wrap streams differently, but we can print the raw text.
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    print(chunk.decode('utf-8'), end='')
            print()
        else:
            print("Response:")
            print(json.dumps(response.json(), indent=2))
            
    except requests.exceptions.ConnectionError:
        print("Failed to connect to the local worker.")
        print("Please ensure the container is running: docker run --gpus all -p 8000:8000 worker-vllm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true", help="Test streaming output")
    args = parser.parse_args()
    
    test_inference(stream=args.stream)
