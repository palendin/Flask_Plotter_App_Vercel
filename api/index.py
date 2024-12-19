from board import create_app

app = create_app()

# The 404 error typically occurs because Vercel needs a specific entry point and structure for Python Flask applications.
# Vercel requires a handler function. 
# Rename run.py to index.py and move it to the api folder.This is a Vercel convention for Serverless Functions
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return app.view_functions[path]() if path in app.view_functions else app.send_static_file(path)

# For local development
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)