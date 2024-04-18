import base64
from PIL import Image
from io import BytesIO

def base64_image_format(base64_string):
    """
    Extracts the image format from a base64 string.

    Args:
        base64_string (str): The base64 string representing an image.

    Returns:
        str: The image format extracted from the base64 string.

    Raises:
        ValueError: If the base64 string is invalid and does not start with 'data:image/{format},'
    """
    if base64_string.startswith('data:image'):
        # Extract the image format from the base64 string
        image_format = base64_string.split(';')[0].split('/')[1]
        return image_format
    else:
        raise ValueError("Invalid base64 string. Must start with 'data:image/{format},'")

def base64_to_image_buffer(base64_string):
    """
    Convert a base64 string representation of an image to an image buffer.

    Args:
        base64_string (str): The base64 string representation of the image.

    Returns:
        BytesIO: The image buffer containing the decoded image bytes.

    Raises:
        ValueError: If the base64 string is invalid and does not start with 'data:image/{format},'
    """
    if base64_string.startswith('data:image'):
        # Remove the image information at the beginning of the base64 string
        base64_data = base64_string.split(',')[1]
    
        # Decode the base64 string into bytes
        image_bytes = base64.b64decode(base64_data)
        
        # Create a BytesIO object to work with the bytes
        image_buffer = BytesIO(image_bytes)
    
        return image_buffer
    else:
        raise ValueError("Invalid base64 string. Must start with 'data:image/{format},'")

def base64_to_image(base64_string):
    """
    Convert a base64 string to an image.

    Args:
        base64_string (str): The base64 string representing the image.

    Returns:
        PIL.Image.Image: The converted image.
    """
    # Convert the base64 string to an image buffer
    image_buffer = base64_to_image_buffer(base64_string)
        
    # Open the image using PIL
    image = Image.open(image_buffer)

    return image


def image_array_to_base64(image_array, format):
    """
    Convert an image array to a base64 string.

    Args:
        image_array (numpy.ndarray): The image array to convert.
        format (str): The format of the image (e.g., 'JPEG', 'PNG').

    Returns:
        str: The base64 string representation of the image.
    """
    # Create a BytesIO object to store the image bytes
    image_buffer = BytesIO()
    
    # Save the image to the buffer in the original format
    image = Image.fromarray(image_array)
    image.save(image_buffer, format=format)
    
    # Get the image bytes from the buffer
    image_bytes = image_buffer.getvalue()
    
    # Encode the image bytes as base64
    base64_data = base64.b64encode(image_bytes).decode('utf-8')
    
    # Add the image information at the beginning of the base64 string
    base64_string = f"data:image/{format.lower()};base64,{base64_data}"
    
    return base64_string

def image_to_base64(image):
    """
    Convert an image to a base64 string.

    Args:
        image (PIL.Image.Image): The image to be converted.

    Returns:
        str: The base64 string representation of the image.
    """
    # Create a BytesIO object to store the image bytes
    image_buffer = BytesIO()
    
    # Save the image to the buffer in the original format
    image.save(image_buffer, format=image.format)
    
    # Get the image bytes from the buffer
    image_bytes = image_buffer.getvalue()
    
    # Encode the image bytes as base64
    base64_data = base64.b64encode(image_bytes).decode('utf-8')
    
    # Add the image information at the beginning of the base64 string
    base64_string = f"data:image/{image.format.lower()};base64,{base64_data}"
    
    return base64_string

def image_file_to_base64(image_filename):
    """
    Convert an image file to a base64 string.

    Args:
        image_filename (str): The path to the image file.

    Returns:
        str: The base64 string representation of the image.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    with Image.open(image_filename) as image:
        # Create a BytesIO object to store the image bytes
        image_buffer = BytesIO()

        # Save the image to the buffer in the original format
        image.save(image_buffer, format=image.format)

        # Get the image bytes from the buffer
        image_bytes = image_buffer.getvalue()

        # Encode the image bytes as base64
        base64_data = base64.b64encode(image_bytes).decode('utf-8')

        # Add the image information at the beginning of the base64 string
        base64_string = f"data:image/{image.format.lower()};base64,{base64_data}"

        return base64_string