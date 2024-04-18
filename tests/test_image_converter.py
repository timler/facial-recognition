import base64
import numpy
from PIL import Image
from io import BytesIO
from watch.image_converter import base64_to_image, image_to_base64, image_file_to_base64, image_array_to_base64

def test_base64_to_image():
    # Prepare test data
    base64_string = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAABQUlEQVQYGQE2Acn+AU9OVQICBAABAgUFBPPy7/fz9BEVGgoMCgcNCAUHBgL//gD//wD+/v/p6OPQz8nt6+UA9vHl3+IB/wIA/wEC//4D/Pr8+vj43drVA//8emBFalYzCP7z+vTz//7/Avv7+/38/vT08//8/DQkGxUSDgYHAzgrEfTt6wIBAAL7+/z6+/vx8OsJBQIpFhT6BhYdHSItNDEA9vH//f0E+/v7AAD//fv9EAYBAwIB9+jh/BMB+ffr//jw//v6Avj7+/r6+wQEAAYGBtjo8fzx8gcYEbG20gwMCAQCAQMWFCL4+vn///8eFxA6NiczOCYSICMRIUAICxj8CRUE+/3+Iic5EA8QKScgDhIL7Qf9BfYUEBkQLRcozs7WAwsLEfX08goND/b39RUaHPT7Be7z/ff4+O7v7uDZ1uTgm8jZvkXdAAAAAElFTkSuQmCC"
    expected_image = Image.open(BytesIO(base64.b64decode(base64_string.split(',')[1])))

    # Call the function
    actual_image = base64_to_image(base64_string)

    # Check the result
    assert actual_image == expected_image

def test_image_to_base64():
    # Prepare test data
    image = Image.open("tests/me_tiny.png")
    expected_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAABMElEQVR4nAXByU7CQBgA4JnpdBERW2QRlcQABokhhoAJPoBXPenBk2d9Aw++kxffwQsS4xYI7iRIg1JaoJ1/Fr8PHx0eE0IRJrpO/cnf3A+c9FosHjOXLN00iJJIKSSl+hl8te9uR24fzbz+2ydWBClMlNQ44ywMe51HTfHLs4Pzk7olfRb4SioCAILLIPAV543SejqZMEytueMEI5dgRAA4A/DG7oJOKpkUMzL5fKHaqKGZp4SgAICQEiBsA2sEzwcffAVHc1eFYwWMhACMAaXIMIyXgce9ibnqXN/cx+MWJZhCpEBECJnZ3EazsVXaK1pO+vTiqvfwPASNgpCF8r69bFfKm4nk4sgU+ixl5+xqdrvVeqLT6fi12y6Wdzs9XnPq77/6MJzoZjSPuPvd/QcSUKAgYFAmjQAAAABJRU5ErkJggg=="

    # Call the function
    actual_base64 = image_to_base64(image)

    # Check the result
    assert actual_base64 == expected_base64

def test_image_file_to_base64():
    # Prepare test data
    image_filename = "tests/me_tiny.png"
    expected_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAABMElEQVR4nAXByU7CQBgA4JnpdBERW2QRlcQABokhhoAJPoBXPenBk2d9Aw++kxffwQsS4xYI7iRIg1JaoJ1/Fr8PHx0eE0IRJrpO/cnf3A+c9FosHjOXLN00iJJIKSSl+hl8te9uR24fzbz+2ydWBClMlNQ44ywMe51HTfHLs4Pzk7olfRb4SioCAILLIPAV543SejqZMEytueMEI5dgRAA4A/DG7oJOKpkUMzL5fKHaqKGZp4SgAICQEiBsA2sEzwcffAVHc1eFYwWMhACMAaXIMIyXgce9ibnqXN/cx+MWJZhCpEBECJnZ3EazsVXaK1pO+vTiqvfwPASNgpCF8r69bFfKm4nk4sgU+ixl5+xqdrvVeqLT6fi12y6Wdzs9XnPq77/6MJzoZjSPuPvd/QcSUKAgYFAmjQAAAABJRU5ErkJggg=="

    # Call the function
    actual_base64 = image_file_to_base64(image_filename)

    # Check the result
    assert actual_base64 == expected_base64

def test_image_array_to_base64():
    # Prepare test data
    image = Image.open("tests/me_tiny.png")
    image_array = numpy.array(image)
    expected_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAABMElEQVR4nAXByU7CQBgA4JnpdBERW2QRlcQABokhhoAJPoBXPenBk2d9Aw++kxffwQsS4xYI7iRIg1JaoJ1/Fr8PHx0eE0IRJrpO/cnf3A+c9FosHjOXLN00iJJIKSSl+hl8te9uR24fzbz+2ydWBClMlNQ44ywMe51HTfHLs4Pzk7olfRb4SioCAILLIPAV543SejqZMEytueMEI5dgRAA4A/DG7oJOKpkUMzL5fKHaqKGZp4SgAICQEiBsA2sEzwcffAVHc1eFYwWMhACMAaXIMIyXgce9ibnqXN/cx+MWJZhCpEBECJnZ3EazsVXaK1pO+vTiqvfwPASNgpCF8r69bFfKm4nk4sgU+ixl5+xqdrvVeqLT6fi12y6Wdzs9XnPq77/6MJzoZjSPuPvd/QcSUKAgYFAmjQAAAABJRU5ErkJggg=="

    # Call the function
    actual_base64 = image_array_to_base64(image_array, "png")

    # Check the result
    assert actual_base64 == expected_base64