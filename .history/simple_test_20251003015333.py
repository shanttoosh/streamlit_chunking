import requests
import pandas as pd
import io

# Create test data
test_data = {
    'name': ['Alice', 'Bob', None, 'Charlie'],
    'age': [25, None, 35, 40],
    'city': ['New York', 'London', 'Paris', None]
}
df = pd.DataFrame(test_data)

csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)
csv_content = csv_buffer.getvalue().encode('utf-8')

try:
    response = requests.post(
        "http://localhost:8000/preview/data",
        files={"file": ("test.csv", csv_content, "text/utf-8")},
        timeout=30
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
