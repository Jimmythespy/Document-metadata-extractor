import cv2 as cv
import numpy as np
import re
import pymupdf
import easyocr as ocr
import pytesseract              # Must install tesseract
import base64
import os
from natsort import natsorted   # sort in the natural order
import requests
import json

from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate

pytesseract.pytesseract.tesseract_cmd = f"D:\\Programs\\tesseract\\tesseract.exe"

openai_key = "key"
antrophic_key = "key"

output_folder = "./result_extracted"    # Extrated image output folder (used for debugging process) - left empty if not needed
scale_source = 2    # Scale the pdf source. The bigger the number the slower the detection will be 
scale_cropped = 4   # Scale the crop image after detection (don't affect detection speed but have lower quality)

llm = ChatOpenAI(
    temperature = 0,
    api_key= openai_key,
    model="gpt-4o",
)

prompt = """
        Instruction:
        \n Extract all texts and numbers.  
        \n Text in the image below is in VietNamese, it's combination of handwritten and typewritten characters.
        \n Text can be in these forms: 
        \n - "Số" + a number (handwritten) + some id string. (take only number and id string)
        \n - A place + date. (take only the date in "day/month/year" format)
        
        \n\n Return only an JSON object with this format: {
            "number" : (number and id string extracted),
            "date" : (date extracted)
        }
        
        \n If any data not mentioned in picture then leave field empty.
    """

def pdf_to_image_preprocessing(pdf_path, crop = 0.3):
    """
        Load the pdf -> return the 1st page as numpy array -> scale it 
        -> crop the page
        
        Args: 
            pdf_path (str): path to the pdf
            crop (float): the percentage of the page to keep from the top
        
        Returns: 
            img (numpy arr): the processed page
    """
    
    scale=scale_source
    
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(0)
    
    mat = pymupdf.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()
    
    top_crop = int(crop * img.shape[0])     # Cropping's y position
    
    return img[:top_crop, :]

def run_tesseract_ocr(image):
    """
        Run OCR on the image
        
        Args: 
            image (np arr): The image needed to OCR
            
        Returns: 
            text (str): The text result from OCR
    """
    
    try: 
        text = pytesseract.image_to_string(image, lang='vie')
        # print(text)
        return text
    except:
        return ""

def pdf_crop_metadata(pdf_path):
    """
        Crop the legal pdf's metadata like
        Purpose: crop out the 
        
        Args: 
            pdf_path: path to pdf file
            
        Returns: 
            List[numpy] list of image detected
    """
    
    image = pdf_to_image_preprocessing(pdf_path)
    
    mid_line = image.shape[1] / 2       # The page mid line position
    buffer = 80                         # The buffer zone in the mid line
    count = 0                           # Output image count
    
    image_list = []

    # Function only for color images
    if image.shape[2] == 4:
        image = cv.cvtColor(image, cv.COLOR_RGBA2BGR)
    else:
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    reader = ocr.Reader(['vi'], gpu=False, model_storage_directory="./model")
    bbox_list = reader.detect(
        image, 
        width_ths=4,            # Threshold of distance to merge text boxes
        mag_ratio=1, 
        min_size=20,
        text_threshold=0.2, 
        slope_ths=0.8,          # For separate texts that have slope
        ycenter_ths =0.5,       # For separate texts that have center offest
        height_ths=0.8,         # Threshold to merge different box height
        add_margin=0.2,
    )

    horizontal_boxes = bbox_list[0][0]      # Get horizontal bb list

    for idx, box in enumerate(horizontal_boxes):
        """
            box format: [x_min, x_max, y_min, y_max]
        """
        left_pos = box[0] - mid_line
        right_pos = box[1] - mid_line
        
        # Check for boxes than cross the document's mid line by a significant amount (more than buffer)
        if ((left_pos < 0 and right_pos < 0) or (left_pos > 0 and right_pos > 0)): 
            pass
        elif (left_pos < 0 and right_pos > 0) or (left_pos > 0 and right_pos < 0):
            if (abs(left_pos) <= buffer and abs(right_pos) > buffer) or (abs(left_pos) > buffer and abs(right_pos) <= buffer):
                pass
            elif (box[3] <= image.shape[0] * 0.5): 
                pass
            else: 
                continue

        text = run_tesseract_ocr(image[box[2]:box[3],box[0]:box[1]])
        if filter_text_result(text):
            res = image_preprocessing(image[box[2]:box[3],box[0]:box[1]], scale=scale_cropped)
            if output_folder != "":
                cv.imwrite(output_folder + f'/image{count+1}.jpg', res)
            count += 1
            image_list.append(res)
    
    return image_list
                
def filter_text_result(text):
    """
        Verify text with correct metadata
        
        Args: 
            text (str): the text need to verify
            
        Returns: 
            bool: True if is verified False if not
    """
    # pattern = r"\d"
    pattern = r'^s|\b(ngày|tháng|năm)\b'

    return bool(re.search(pattern, text, re.IGNORECASE))

def image_preprocessing(image, scale=10):
    """
        Open image then preprocess the image using image processing
        (Empty since process the image doesn't improve the accuracy 
        of the OCR result of the LLM. This step only needed if 
        the image is feed into an deep learning model)
        
        Args: 
            image_name (numpy arr): image to be processed
            scale (int): scale factor on the cropped image 
        
        Returns: 
            image in numpy arr format
    """
    
    # stretch_num = 80
    
    # Resize and stretch image in x direction
    # image = cv.resize(image, (image.shape[1] + stretch_num, image.shape[0]), interpolation=cv.INTER_LANCZOS4)
    image = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)

    # Gray scale and thresholding
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # _, image = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    
    # Sharpening
    # blurred = cv.GaussianBlur(image, (9, 9), 8.0)
    # image = cv.addWeighted(image, 2, blurred, -0.5, 0)
    
    return image

def image_encoder(image):
    """
        Loads an image from the specified path and converts it to a base64 string.
        
        Args: 
            image (np arr): numpy image
            
        Returns: 
            str: Base64-encoded string of the image.
    """
    
    _, buffer = cv.imencode(".jpg", image)
    base64_str = base64.b64encode(buffer).decode("utf-8")
    return base64_str

def llm_ocr_chatGPT(image_list):
    """
        Run OCR on extracted pdf's metadata using chatGPT
        
        Args: 
            image_list (np arr): list of numpy array images
            
        Returns: 
            Str: extracted content from LLM in markdown format
    """
    global prompt
        
    prompt_content = [
        {
            "type" : "text",
            "text" : prompt
        }
    ]
    
    for image in image_list:
        image = image_encoder(image)
        
        prompt_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}",
                    "detail": "high"
                },
            }
        )
    
    prompt_OCR = ChatPromptTemplate([
        HumanMessage(content=prompt_content)
    ])

    chain = prompt_OCR | llm

    return chain.invoke({"input" : ""}).content

def llm_ocr_claude(image_list):
    """
        Run OCR on extracted pdf's metadata using Claude
        
        Args: 
            image_list (np arr): list of numpy array images
            
        Returns: 
            Str: extracted content from LLM in markdown format
    """
    
    global prompt
    
    payload_content = [
        {
            "type": "text",
            "text": prompt
        },
    ]
    
    for image in image_list:
        image_base64 = image_encoder(image)
        
        payload_content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",  # Adjust based on your image type
                    "data": image_base64
                }
            }
        )
    
    # Prepare the API request
    headers = {
        "x-api-key": antrophic_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # Construct the message with text and image content
    messages = [
        {
            "role": "user",
            "content": payload_content
        }
    ]
    
    # Prepare the payload
    payload = {
        "model": "claude-3-7-sonnet-20250219",  # Use the desired Claude model
        "messages": messages,
        "max_tokens": 1000
    }
    
    # Make the API call
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload
    )
    
    # Return the response
    return response.json()["content"][0]["text"]

def clean_llm_output(res):
    """Convert LLM markdown output to python object"""
    
    match = re.search(r"\{(.*)\}", res, re.DOTALL)
    if match:
        cleaned = match.group(0)
    else: 
        cleaned = """{"number": "", "date" : ""}"""

    return json.loads(cleaned)

def ocr_multiple_pdfs(pdfs_path, output_file_name, LLM="chatGPT"):
    """
        Test OCR on multiple pdf using different LLM options
        
        Args: 
            pdfs_path (str): path to the folder contain all the pdf docs
            output_path (str) : output file name
            model (str): model options ["chatGPT", "claude"]
        
        Returns: 
            str : LLM result
    """

    pdf_list = os.listdir(pdfs_path)
    pdf_list_sorted = natsorted(pdf_list)

    string_result = ""

    for pdf in pdf_list_sorted:
        list = pdf_crop_metadata(pdfs_path + "/" + pdf)
        
        if LLM == "chatGPT":
            res = llm_ocr_chatGPT(list)
        if LLM == "claude": 
            res = llm_ocr_claude(list)
        else: 
            raise Exception("LLM options must be correct") 

        cleaned = clean_llm_output(res)
        
        print(pdf)
        print(cleaned)
        
        string_result += cleaned["number"] + "     " + cleaned["date"] + "     " + pdf + "\n"

    with open(output_file_name, "w", encoding="utf-8") as file:
        try:
            file.write(string_result)
        except Exception as e: 
            print(e)

if __name__ == "__main__":
    # Use on single pdfs
    # list = pdf_crop_metadata("./docs_pdf/44cc52f89c9b41c78aa8b2c4b262ed00.pdf")
    # res = clean_llm_output(llm_ocr_claude(list))
    
    # Use on multiple pdfs
    ocr_multiple_pdfs("./docs_pdf/", "result_claude.text", "claude")