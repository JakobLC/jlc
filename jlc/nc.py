import numpy as np
import matplotlib.colors as mcolors

matplotlib_pallete =   [0,0,0,
                        31, 119, 180, 
                        255, 127, 14, 
                        44, 160, 44, 
                        214, 39, 40, 
                        148, 103, 189, 
                        140, 86, 75, 
                        227, 119, 194, 
                        127, 127, 127, 
                        188, 189, 34, 
                        23, 190, 207]

cityscapes_pallete =   [0, 0, 0,
                        128, 64, 128,
                        244, 35, 232,
                        70, 70, 70,
                        102, 102, 156,
                        190, 153, 153,
                        153, 153, 153,
                        250, 170, 30,
                        220, 220, 0,
                        107, 142, 35,
                        152, 251, 152,
                        0, 130, 180,
                        220, 20, 60,
                        255, 0, 0,
                        0, 0, 142,
                        0, 0, 70,
                        0, 60, 100,
                        0, 80, 100,
                        0, 0, 230,
                        119, 11, 32]


large_pallete = [  0,   0,   0,  23, 190, 207, 255, 127,  14, 214,  39,  40, 152,
       251, 152,   0,   0, 142, 148, 103, 189, 220, 220,   0, 140,  86,
        75, 107, 142,  35, 220,  20,  60, 255,   0,   0, 255, 255,  90,
       102, 102, 156,  31, 119, 180,   0,   0,  70, 119,  11,  32, 205,
       255,  50,   0,  80, 100, 250, 170,  30,   0,   0, 230, 244,  35,
       232, 227, 119, 194, 255, 220,  80,  44, 160,  44, 190, 153, 153,
       128,  64, 128,   0,  60, 100]

largest_pallete = [  0,   0,   0]+sum([large_pallete[3:] for _ in range(255*3//len(large_pallete)+2)],[])[:255*3]

matplotlib_colors = np.array(matplotlib_pallete[3:]).reshape(-1, 3)
cityscapes_colors = np.array(cityscapes_pallete[3:]).reshape(-1, 3)
large_colors = np.array(large_pallete[3:]).reshape(-1, 3)
largest_colors = np.array(largest_pallete[3:]).reshape(-1, 3)

def convert_mpl_color(color, output_type="tuple"):
    """
    Convert a color to the desired output format.

    Parameters:
    color : str, tuple, list
        The input color which can be a color name, hex code, or a tuple/list of RGB(A) values.
    output_type : str
        The desired output format: "tuple", "list", "hex", or "name".

    Returns:
    converted_color : tuple, list, str
        The color converted to the desired format.
    """
    
    # Convert the color to RGBA format first
    try:
        rgba = mcolors.to_rgba(color)
    except ValueError:
        raise ValueError("Invalid color format")

    # Convert to the desired output type
    if output_type == "tuple":
        return rgba
    elif output_type == "list":
        return list(rgba)
    elif output_type == "hex":
        return mcolors.to_hex(rgba)
    elif output_type == "name":
        if color in mcolors.CSS4_COLORS:
            return color
        else:
            # Attempt to find the closest named color
            name, min_dist = None, float('inf')
            for cname, crgba in mcolors.CSS4_COLORS.items():
                dist = sum((c1 - c2) ** 2 for c1, c2 in zip(rgba, mcolors.to_rgba(crgba)))
                if dist < min_dist:
                    name, min_dist = cname, dist
            return name
    else:
        raise ValueError("Invalid output type")