from django.shortcuts import render
from .predictor import predict_pneumonia

def front_page(request):
    prediction = None
    if request.method == 'POST' and request.FILES.get('scan'):
        image_file = request.FILES['scan']
        try:
            prediction = predict_pneumonia(image_file)
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render(request, 'scan/front.html', {'prediction': prediction})