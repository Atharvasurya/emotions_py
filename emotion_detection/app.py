from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# List of possible emotions
emotions = ['Happy', 'Sad', 'Angry', 'Neutral', 'Surprised']

def generate_frames():
    camera = cv2.VideoCapture(0)  # Use default camera
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Process each face
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Simulate emotion detection (random for demonstration)
                emotion = np.random.choice(emotions)
                
                # Display emotion text
                cv2.putText(frame, emotion, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Convert frame to bytes
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield the frame in bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Real-time Emotion Detection</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f0f0; }
            .container { max-width: 800px; margin: 0 auto; text-align: center; }
            h1 { color: #333; }
            .video-container { margin: 20px 0; background: white; padding: 10px; border-radius: 8px; }
            img { max-width: 100%; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Real-time Emotion Detection</h1>
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" alt="Video feed">
            </div>
            <p>Note: This is a demonstration using simulated emotions.</p>
        </div>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True) 