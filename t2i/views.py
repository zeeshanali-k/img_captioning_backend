# from django.http import JsonResponse
# from transformers import AutoProcessor
# import torch
# from PIL import Image


# def index(request):

#     processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
#     model = torch.load(
#         "/Users/zeeshanali/Documents/ML Projects/image2Text/i2t_backend/t2i/model.pt",
#         map_location=torch.device("cpu"),
#     )
#     model.eval()
#     idx = 6
#     img = Image.open(
#         "/Users/zeeshanali/Documents/ML Projects/image2Text/i2t_backend/t2i/badshahi mosque (21).jpg"
#     ).convert("RGB")

#     # device = "cuda" if torch.cuda.is_available() else "cpu"
#     inputs = processor(images=img, return_tensors="pt").to("cpu")
#     pixel_values = inputs.pixel_values

#     generated_ids = model.generate(pixel_values=pixel_values, max_length=128)
#     generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[
#         0
#     ]

#     data = {"caption": generated_caption}
#     return JsonResponse(data)

from django.http import JsonResponse
from django.core.files.uploadedfile import UploadedFile

from transformers import AutoProcessor
import torch
from PIL import Image

from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def index(request):
    print("Captioning Image")
    if request.method == "POST":
        # Check for uploaded image
        if "image" not in request.FILES:
            return JsonResponse({"error": "No image uploaded."})

        image_file: UploadedFile = request.FILES["image"]

        # Process the uploaded image
        try:
            img = Image.open(image_file).convert("RGB")  # Open and convert to RGB
        except Exception as e:
            return JsonResponse({"error": f"Error processing image: {str(e)}"})

        processor = AutoProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        model = torch.load(
            "/Users/zeeshanali/Documents/ML Projects/image2Text/i2t_backend/t2i/model.pt",
            map_location=torch.device("cpu"),
        )
        model.eval()

        # device = "cuda" if torch.cuda.is_available() else "cpu"  # Commented out for efficiency
        inputs = processor(images=img, return_tensors="pt").to("cpu")
        pixel_values = inputs.pixel_values

        generated_ids = model.generate(pixel_values=pixel_values, max_length=128)
        generated_caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        data = {"caption": generated_caption}
        return JsonResponse(data)

    else:
        # Handle other HTTP methods (e.g., GET) as needed
        return JsonResponse({"message": "Please upload an image using POST request."})
