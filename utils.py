def create_subimages(image, rows, cols):
    height, width = image.shape
    subimage_width, subimage_height = int(width / cols), int(height / rows)
    subimages, coords = [], []

    for y in range(0, height, subimage_height):
        for x in range(0, width, subimage_width):
            subimage = image[y:y + subimage_height, x:x + subimage_width]

            subimages.append(subimage)
            coords.append((x, y))

    return subimages, coords