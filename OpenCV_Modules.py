# Dependencies
import numpy as np
import matplotlib.pyplot as plt
import cv2


# We can analyze an image or a video in opencv.


# Define function to load image, with specified color scheme
def LoadImage(im_name, color=1):
	# This loads the image as color image, but without alpha channel (opaquness)
	# Channel sequence is BGR-A
	if color == 1:
		im = cv2.imread(im_name, cv2.IMREAD_COLOR)
	# This loads the image as grayscaled
	# Decreases image processing time
	elif color == 0:
		im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)
	# Unchanged load. The second argument can also be replaced by the number itself. e.g. im = cv2.imread(im_name, -1)
	elif color == -1:
		im = cv2.imread(im_name, cv2.IMREAD_UNCHANGED)
	# Default case!
	else :
		im = cv2.imread(im_name)
	return im


# Define a function to display the image
def DisplayImage(im, window_name):
	# Display the image in a named window
	cv2.namedWindow(str(window_name), cv2.WINDOW_NORMAL)
	cv2.imshow(str(window_name), im)
	cv2.waitKey(0) # Wait for any key to press, we can keep this to be the q key as well!!
	cv2.destroyAllWindows()


# Define a function to display the image with matplotlib
def PlotImage(im, color):
	fig = plt.figure()
	# Grayscale plot
	if color == 'gray':
		plt.imshow(im, cmap = 'gray', interpolation = 'bicubic')
		plt.axis('off')
		plt.show()
	cv2.waitKey(0) # Wait for a key to press
	plt.close(fig)


# Define a function that adds a simple line with matplotlib
def AddLineToImage(im, point1, point2):
	fig = plt.figure()
	# Assume that the point1, point2 entries are in [x, y] format
	plt.imshow(im, cmap = 'gray', interpolation = 'bicubic')
	plt.plot(point1, point2, 'b', linewidth = 5)
	plt.show()
	cv2.waitKey(0)
	plt.close(fig)


# Define a function to save an image
def SaveImage(im, im_name):
	# Assumes that the image has appropriate extensions and data
	cv2.imwrite(str(im_name), im)


# Define a function to capture video
def CaptureVideo( vid_name, which_webcam = 0):
	# 0 means the first webcam on the system, 1 means second...
	capt_vid = cv2.VideoCapture(int(which_webcam))
	# Loop for returned frames
	while 1:
		# If there is a feed, there is_ret! Frame is the returned frame
		# We can feed multiple images
		is_ret, frame = capt_vid.read()
		#gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#true_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # THIS FLIPS THE COLORS!! TRAP
		cv2.imshow(str(vid_name), frame)
		#cv2.imshow(str(vid_name) + 'Gray', gray_frame)
		#cv2.imshow(str(vid_name) + 'Color_Not_Orig', true_frame)
		# Define when to break
		if cv2.waitKey(1) and 0xFF == ord('q'): # At key q, ...
			break
	# Release the feed!
	capt_vid.release()
	cv2.destroyAllWindows()


# Define a function to write a video file
def WriteVideo(out_name, out_size, which_webcam = 0, is_display = 1):
	# Replace the which_webcam in the next line with the name of the video file to load a video file!
	capt_vid = cv2.VideoCapture(int(which_webcam))
	vid_codec = cv2.VideoWriter_fourcc(*'XVID') # Works for all systems (??)
	output = cv2.VideoWriter(str(out_name), vid_codec, 20, out_size) # Last is the output size. e.g. (640, 480)
	# Save the frames
	while 1:
		is_ret, frame = capt_vid.read()
		output.write(frame)
		if is_display:
			cv2.imshow('Video_Write', frame)
		if cv2.waitKey(25):
			break
	# Release the output and feed
	capt_vid.release()
	output.release()


# Define a function to draw a line on an image
def DrawLineOnImage(im_name, point1, point2, line_color = (0, 255, 0), linewidth = '10', window_name = 'Image', im_color = 1):
	# Load the image
	im = cv2.imread(str(im_name), im_color) # OR, cv2.IMREAD_COLOR/GRAYSCALE/UNCHANGED
	# The points are assumed (x, y) tuples
	# The line_color is (r, g, b) tuple with 0-255 range black-white. Note that it is BGR, and not RGB!
	# The linewidth is an integer
	# Add line
	cv2.line(im, point1, point2, line_color, int(linewidth))
	# Display the image
	cv2.imshow(str(window_name), im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# Define a function to draw a rectangle on an image
def DrawRectangleOnImage(im_name, point1, point2, line_color = (0, 255, 0), linewidth = '10', window_name = 'Image', im_color = 1):
	# Load the image
	im = cv2.imread(str(im_name), im_color) # OR, cv2.IMREAD_COLOR/GRAYSCALE/UNCHANGED
	# The points are assumed (x, y) tuples with point1 being top left and point2 being bottom right
	# The line_color is (r, g, b) tuple with 0-255 range black-white. Note that it is BGR, and not RGB!
	# The linewidth is an integer
	# Add line
	cv2.rectangle(im, point1, point2, line_color, int(linewidth))
	# Display the image
	cv2.imshow(str(window_name), im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# Define a function to draw a circle on an image
def DrawCircleOnImage(im_name, center, radius, fill = 0, line_color = (0, 255, 0), linewidth = '10', window_name = 'Image', im_color = 1):
	im = cv2.imread(str(im_name), im_color)
	# Center is (x, y) tuple
	# Fill is an int. If fill = -1, the circle is filled.
	cv2.circle(im, center, radius, line_color, fill)
	cv2.imshow(str(window_name), im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# Define a function to plot a polygon on an image
def DrawPolygonOnImage(im_name, vertices, is_connect = True, line_color = (0, 255, 0), linewidth = '10', window_name = 'Image', im_color = 1):
	im = cv2.imread(str(im_name), im_color)
	# is_connect tells if we want to connect the last point to the first
	cv2.polylines(im, [vertices], is_connect, line_color, int(linewidth)) 
	cv2.imshow(str(window_name), im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# Define a function to add text to an image
def DrawTextOnImage(im_name, text, start_point = (0, 0), font = cv2.FONT_HERSHEY_SIMPLEX, size = 1, line_color = (0, 255, 0), spacing = 2, linewidth = '10', window_name = 'Image', im_color = 1):
	im = cv2.imread(str(im_name), im_color)
	cv2.putText(im, str(text), start_point, font, int(size), line_color, spacing, cv2.LINE_AA)
	cv2.imshow(str(window_name), im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# Define a function to weighted add two images
def WeightedAdd(im1_name, weight1, im2_name, weight2, gamma = 0, im_color = 1, window_name = 'Image'):
	im1 = cv2.imread(str(im1_name), im_color)
	im2 = cv2.imread(str(im2_name), im_color)
	# im1 + im2 can be done. It retains the image opaqueness
	# add adds the images pixel by pixel and then clip the values at 255
	im = cv2.addWeighted(im1, weight1, im2, weight2, gamma)
	cv2.namedWindow(str(window_name), cv2.WINDOW_NORMAL)
	cv2.imshow(str(window_name), im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()





# Main
if __name__ == "__main__":
	##################################################
	#im = LoadImage('Me.jpg', -1)
	#DisplayImage(im, 'Eeshan')
	##################################################
	#im = LoadImage('Me.jpg', 0)
	#PlotImage(im, 'gray')
	##################################################
	#im = LoadImage('Me.jpg', 0) # Fix the color scheme appropriately BEFOREHAND
	#AddLineToImage(im, [10, 20], [120, 90])
	##################################################	
	#CaptureVideo('Eeshan_Video', 0)
	##################################################
	#WriteVideo('Eeshan_Movie.avi', (640, 480), 0, 0)	
	##################################################
	#DrawLineOnImage('Me.jpg', (10, 20), (50, 5))
	##################################################
	#DrawRectangleOnImage('Me.jpg', (10, 20), (50, 5)) # This also gives a rect!
	#DrawRectangleOnImage('Me.jpg', (10, 20), (50, 60))
	##################################################
	#DrawCircleOnImage('Me.jpg', (150, 100), 30, fill = -1) # Filled circle
	#DrawCircleOnImage('Me.jpg', (150, 100), 30, fill = 0) # Hollow circle
	##################################################
	#vertices = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
	#print vertices.shape # It is (4, 2)
	#vertices = vertices.reshape((-1, 1, 2)) # Not required
	#print vertices.shape # It is (4, 1, 2) # Not required
	#DrawPolygonOnImage('Me.jpg', vertices, is_connect = True)
	##################################################
	#DrawTextOnImage('Me.jpg', 'Eeshan', start_point = (20, 100))
	##################################################
	#WeightedAdd('Me.jpg', 0.8, 'Astro.jpg', 0.2, gamma = 0, im_color = 1, window_name = 'Image')