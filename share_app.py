import os
import sys
import subprocess
import time
from pyngrok import ngrok

# Start the Streamlit app in the background
print("Starting Streamlit app...")
streamlit_process = subprocess.Popen([sys.executable, "run_streamlit.py"])

# Give Streamlit a moment to start
time.sleep(5)

# Set up ngrok tunnel (Streamlit uses port 8501 by default)
try:
    # You can set your ngrok auth token here if you have one
    # ngrok.set_auth_token("YOUR_AUTH_TOKEN")
    
    http_tunnel = ngrok.connect(8501)
    public_url = http_tunnel.public_url
    print(f"\n🔗 Your app is now available at: {public_url}")
    print("Share this URL with your friends so they can try your app!")
    print("\nKeep this window open while they're using the app.")
    print("Press Ctrl+C when you're done to shut down the server.\n")
    
    # Keep the script running until manually interrupted
    while True:
        time.sleep(1)
        
except KeyboardInterrupt:
    print("Shutting down the server...")
    streamlit_process.terminate()
    ngrok.kill()
    print("Server has been shut down.") 