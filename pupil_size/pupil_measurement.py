import cv2 
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction 

def detect_eyes_in_color_image(image):
    """Detect eyes in a color image using Haar cascades"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load eye cascade classifier
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(eyes) == 0:
        # Try face detection and then look for eyes in face region
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Take the first face and look for eyes within it
            (fx, fy, fw, fh) = faces[0]
            face_roi = gray[fy:fy+fh, fx:fx+fw]
            eyes_in_face = eye_cascade.detectMultiScale(face_roi, 1.1, 4)
            
            if len(eyes_in_face) > 0:
                # Convert coordinates back to full image
                ex, ey, ew, eh = eyes_in_face[0]
                return fx + ex + ew//2, fy + ey + eh//2
    
    if len(eyes) > 0:
        # Return center of first detected eye
        (ex, ey, ew, eh) = eyes[0]
        return ex + ew//2, ey + eh//2
    
    # Fallback: return center of image
    h, w = image.shape[:2]
    return w//2, h//2

def get_eye_bounding_box(image, eye_x, eye_y, offset=80):
    """Extract bounding box around eye region"""
    validation = image.copy()
    return validation[eye_y - offset : eye_y + offset, eye_x - offset : eye_x + offset], eye_x - offset, eye_y - offset

def yiq_conversion(rgb_vector):
    """Convert RGB to YIQ color space"""
    height, width, _ = np.shape(rgb_vector)

    rgb_vector = np.asarray(rgb_vector)
    yiq_matrix = np.array([[0.299  ,  0.587  , 0.114   ],
                           [0.5959 , -0.2746 , -0.3213 ],
                           [0.2115 , -0.5227 , 0.3112  ]])
    b = rgb_vector[:, :, 0]
    g = rgb_vector[:, :, 1]
    r = rgb_vector[:, :, 2]

    for x in range(0, width):
        for y in range(0,height):
            rgb_vector[x][y] = yiq_matrix.dot([b[x,y],g[x,y],r[x,y]])
    return rgb_vector

def median_color_of_image(image):
    """Calculate median color of image"""
    mask = image.copy()
    median_array = np.asarray(mask)
    sort = np.sort(median_array)
    medians = np.median(sort, axis=0)
    return medians[len(medians) // 2]

def draw_median_circle(image, eye_x, eye_y, median):
    """Draw a circle with median color at eye position"""
    median_image = image.copy()
    cv2.circle(median_image, (eye_x, eye_y), 40, (int(median[0]), int(median[2]), int(median[1])), -1)
    return median_image

def validate_region(image):
    """Validate if the region looks like an eye area"""
    mean, std = cv2.meanStdDev(image)

    if mean[1] > 200 and mean[0] < 100 and mean[2] < 50:
        message = "The surrounding region is dominated by skin, therefore it has a good chance of being the eye"
        return True, mean, std, message
    else:
        message = "Region is not dominant by skin therefore poor chance of being eye region"
        return False, mean, std, message

def validator(image, eye_x, eye_y):
    """Validate the detected eye region"""
    original = image.copy()

    bounded_image, cropped_x, cropped_y = get_eye_bounding_box(original, eye_x, eye_y)

    yiq_image = yiq_conversion(bounded_image)

    median_color = median_color_of_image(yiq_image)

    bounded = draw_median_circle(yiq_image, eye_x - cropped_x, eye_y - cropped_y, median_color)

    validation, mean, std, message = validate_region(bounded)

    return validation, mean, std, message, yiq_image

def crop_to_eye(image, eye_x, eye_y):
    """Crop image to 250x250 region centered on the eye"""
    return image[eye_y-125:eye_y+125, eye_x-125:eye_x+125]

def blur_grayscale(image, blur):
    """Convert to grayscale and apply blur for Circle Hough Transform"""
    blurred_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.medianBlur(blurred_image, blur)

def draw_detection(image, detection):
    """Draw detected circles on image"""
    try:
        detection = np.uint16(np.around(detection))
        x, y, r = detection[0, 0]
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        cv2.circle(image, (x, y), r, (255, 0, 0), 1)
        return image, x, y, r
    except:
        print("No circles were found")
        return image, 0, 0, 0

def get_ratio(pr, ir):
    """Calculate the ratio and convert to millimeters"""
    # Radius in millimeters
    iris_mm = 6.5

    if ir == 0:  # Avoid division by zero
        return 0, 0, 0, iris_mm

    ratio = Fraction(pr, ir)
    pupil_px = ratio.numerator
    iris_px = ratio.denominator

    # Using the ratio of millimeters : pixels for the iris,
    # we can use this factor to find the size in mm of the pupil
    # since we know its size in pixels
    px_to_mm_factor = iris_mm / iris_px
    pupil_mm = pr * px_to_mm_factor

    return pupil_px, iris_px, pupil_mm, iris_mm

def detect_iris_and_pupil(image, eye_x, eye_y):
    """Detect iris and pupil in the image"""
    if image is None:
        print("Image is empty")
        return None, 0, 0, 0, 0, None
    
    # Crop down to the eye region
    cropped_image = crop_to_eye(image, eye_x, eye_y)

    # Use one image for detecting the iris, pupil
    iris_detection = cropped_image.copy()
    pupil_detection = cropped_image.copy()
    final_detection = cropped_image.copy()

    # Find the iris
    gray_cropped = blur_grayscale(iris_detection, 17)
    iris = cv2.HoughCircles(gray_cropped, cv2.HOUGH_GRADIENT, 1, iris_detection.shape[0], param1=50, param2=20, minRadius=0, maxRadius=48)
    iris_detection, ix, iy, ir = draw_detection(iris_detection, iris)

    # Find the pupil
    gray_cropped = blur_grayscale(pupil_detection, 19)
    pupil = cv2.HoughCircles(gray_cropped, cv2.HOUGH_GRADIENT, 1, pupil_detection.shape[0], param1=50, param2=15, minRadius=0, maxRadius=max(ir-4, 0))
    pupil_detection, px, py, pr = draw_detection(pupil_detection, pupil)

    # Draw the final detection
    if ir > 0:
        cv2.circle(final_detection, (ix, iy), 2, (0, 0, 255), 2)
        cv2.circle(final_detection, (ix, iy), ir, (255, 0, 0), 2)
    if pr > 0:
        cv2.circle(final_detection, (px, py), pr, (255, 0, 0), 2)

    # Get ratio in pixels and millimeters
    pupil_px, iris_px, pupil_mm, iris_mm = get_ratio(pr, ir)

    return final_detection, pupil_px, iris_px, pupil_mm, iris_mm, gray_cropped

def display_results(color_image, yiq_image, mean, cropped_eye, grey_cropped, final_detection, eye_x, eye_y, message, pupil_px, iris_px, pupil_mm, iris_mm):
    """Display the analysis results"""
    plt.figure(figsize=(20, 15))
    grid = plt.GridSpec(3, 3)  
    plt.suptitle("Pupil Size Analysis", fontsize=16)
    
    plt.subplot(grid[0,0])
    plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Color Image')
    plt.axis('off')

    plt.subplot(grid[0,1])
    plt.imshow(cv2.cvtColor(yiq_image, cv2.COLOR_BGR2RGB))
    plt.title('YIQ of Eye Region')
    plt.axis('off')

    plt.subplot(grid[0,2])
    # Bar chart of the mean colors
    names = ["Blue", "Green", "Red"]
    values = np.ravel(mean)
    plt.bar(names, values, color=["blue", "green", "red"])
    plt.title('Mean Color Values')
    plt.ylabel('Color Scale')

    plt.subplot(grid[1,0])
    plt.imshow(cv2.cvtColor(cropped_eye, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Eye Region')
    plt.axis('off')

    plt.subplot(grid[1,1])
    plt.imshow(grey_cropped, cmap='gray')
    plt.title('Grayscale Eye')
    plt.axis('off')

    plt.subplot(grid[1,2])
    plt.imshow(cv2.cvtColor(final_detection, cv2.COLOR_BGR2RGB))
    plt.title('Pupil & Iris Detection')
    plt.axis('off')

    plt.subplot(grid[2, :])
    plt.axis("off")
    result_text = f'''
    Eye Coordinates: X={eye_x}, Y={eye_y}
    {message}
    
    Pupil to Iris Ratio:
    - In pixels: {pupil_px}px : {iris_px}px
    - In millimeters: {pupil_mm:.2f}mm : {iris_mm}mm
    
    Pupil Size: {pupil_mm:.2f}mm
    '''
    plt.text(0.1, 0.5, result_text, fontsize=12, verticalalignment='center',
             bbox={'facecolor': 'lightblue', 'alpha': 0.8, 'pad': 20})
    
    plt.tight_layout()
    plt.show()

def measure_pupil_from_image(image_path):
    """Main function to measure pupil size from a single color image"""
    # Load the image
    color_image = cv2.imread(image_path, 1)
    
    if color_image is None:
        print(f"Could not load image: {image_path}")
        return False
    
    print("Detecting eyes in the image...")
    # Detect eye coordinates
    eye_x, eye_y = detect_eyes_in_color_image(color_image)
    print(f"Eye detected at coordinates: ({eye_x}, {eye_y})")
    
    # Validate the eye region
    validation, mean, std, message, yiq_image = validator(color_image, eye_x, eye_y)
    
    if not validation:
        print("Warning: Eye region validation failed, but continuing anyway...")
        print(message)
    
    # Crop eye region for display
    cropped_eye, _, _ = get_eye_bounding_box(color_image, eye_x, eye_y)
    
    # Detect iris and pupil
    print("Detecting iris and pupil...")
    final_detection, pupil_px, iris_px, pupil_mm, iris_mm, grey_cropped = detect_iris_and_pupil(color_image, eye_x, eye_y)
    
    if final_detection is None:
        print("Failed to detect pupil and iris")
        return False
    
    # Display results
    display_results(color_image, yiq_image, mean, cropped_eye, grey_cropped, 
                   final_detection, eye_x, eye_y, message, pupil_px, iris_px, pupil_mm, iris_mm)
    
    return True

# Example usage
if __name__ == "__main__":
    # You can now just call this with a single image
    measure_pupil_from_image("2019-11-22-17-33-27_Color.jpeg")