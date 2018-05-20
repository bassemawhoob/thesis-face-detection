def pyramid(image, scale=1.25, minSize=(19, 19)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.width/scale)
        h = int(image.height / scale)
        image = image.resize((w,h))

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.height < minSize[1] or image.width < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.height, stepSize):
        for x in range(0, image.width, stepSize):
            # yield the current window
            # yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            yield (x,y, image.crop((x,y,int(x+windowSize[0]),int(y+windowSize[1]))))
