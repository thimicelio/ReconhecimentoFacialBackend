# Facial Recognition Backend

This project is an independent API developed for an academic assignment at
UEMG, during the 7th semester of the Computer Engineering course.

The API receives a person's photo, a document photo, and a similarity threshold.
It compares the face in both images and also extracts document information using
OCR. The result is returned as JSON.

## Technologies

This project uses Python 3.11.9.

Dependencies listed in `requirements.txt`:

```txt
numpy==1.26.4
flask==3.0.3
paddlepaddle==2.6.2
paddleocr==2.7.3
opencv-python
scikit-learn==1.5.2
```

## How It Works

The API exposes a POST endpoint that receives:

- `img_person`: image of the person.
- `img_doc`: image of the document.
- `threshold`: minimum similarity value between `0` and `1`.

The backend performs two main tasks:

1. It uses OpenCV to find and compare the face from `img_person` with the face
   found in `img_doc`.
2. It uses PaddleOCR to read the document image and extract fields such as name,
   CPF, RG, and dates when possible.

The response contains the extracted document data and whether both images appear
to be from the same person.

## Running the API

Create and activate a Python 3.11.9 virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the API:

```powershell
python api.py
```

By default, the API runs at:

```txt
http://127.0.0.1:5000
```

Available endpoints:

```txt
POST /api/validate-document
POST /api/images
```

## Example Response

```json
{
  "same_person": true,
  "similarity": 0.78,
  "threshold": 0.7,
  "document": {
    "nome": "Example Name",
    "cpf": "000.000.000-00",
    "rg": "MG0000000",
    "datas": {
      "nascimento": "01/01/2000",
      "validade": "01/01/2030"
    }
  }
}
```

## Frontend Example

There is also a simple usage example available in the `frontend` branch. That
branch contains only static HTML, CSS, and JavaScript files that demonstrate how
to send the images and threshold parameter to this API.
