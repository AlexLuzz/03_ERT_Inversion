import fitz  # PyMuPDF
from PIL import Image
import imageio
import io

def pdf_to_gif(pdf_path, gif_path, duration=0.5, zoom_factor=1):
    """
    Convert a PDF into a GIF with one frame per page.
    
    Parameters:
    - pdf_path: The file path of the input PDF.
    - gif_path: The file path to save the output GIF.
    - duration: Duration for each frame in the GIF (in seconds).
    """
    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    # List to store PIL images
    images = []

    # Iterate over each page of the PDF
    for page_num in range(17, pdf_document.page_count-6):
        page = pdf_document.load_page(page_num)
        
        # Create a transformation matrix with the specified zoom factor
        matrix = fitz.Matrix(zoom_factor, zoom_factor)  # Increase resolution to avoid cropping
        
        # Render the page to an image (pixmap) with the matrix transformation
        pix = page.get_pixmap(matrix=matrix)

        # Convert the pixmap to a PIL Image
        img = Image.open(io.BytesIO(pix.tobytes()))
        
        # Append the image to the list
        images.append(img)

    # Create a GIF from the list of images
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration * 1000, loop=0)

    print(f"GIF created and saved as {gif_path}")

# Example Usage
pdf_path = 'D:/02_ERT_Data/11-20_18h_11-26_13h_ratios_TL.pdf'  # Path to the input PDF
gif_path = 'D:/02_ERT_Data/RE_gif.gif'   # Path to save the resulting GIF

pdf_to_gif(pdf_path, gif_path)
