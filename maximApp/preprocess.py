def mod_padding_symmetric(image, factor=64):
    """Padding the image to be divided by factor."""
    height, width = image.shape[0], image.shape[1]
    height_pad, width_pad = ((height + factor) // factor) * factor, (
        (width + factor) // factor
    ) * factor
    padh = height_pad - height if height % factor != 0 else 0
    padw = width_pad - width if width % factor != 0 else 0
    image = tf.pad(
        image, [(padh // 2, padh // 2), (padw // 2, padw // 2), (0, 0)], mode="REFLECT"
    )
    return image


def make_shape_even(image):
    """Pad the image to have even shapes."""
    height, width = image.shape[0], image.shape[1]
    padh = 1 if height % 2 != 0 else 0
    padw = 1 if width % 2 != 0 else 0
    image = tf.pad(image, [(0, padh), (0, padw), (0, 0)], mode="REFLECT")
    return image


def process_image(image: Image):
    input_img = np.asarray(image) / 255.0
    height, width = input_img.shape[0], input_img.shape[1]

    # Padding images to have even shapes
    input_img = make_shape_even(input_img)
    height_even, width_even = input_img.shape[0], input_img.shape[1]

    # padding images to be multiplies of 64
    input_img = mod_padding_symmetric(input_img, factor=64)
    input_img = tf.expand_dims(input_img, axis=0)
    return input_img, height, width, height_even, width_even


def init_new_model(input_img):
    variant = ckpt.split("/")[-1].split("_")[0]
    print(variant)
    configs = MAXIM_CONFIGS['S-2']
    print(configs)
    configs.update(
        {
            "variant": "S-2",
            "dropout_rate": 0.0,
            "num_outputs": 3,
            "use_bias": True,
            "num_supervision_scales": 3,
        }
    )  # From https://github.com/google-research/maxim/blob/main/maxim/run_eval.py#L45-#L61
    configs.update({"input_resolution": (input_img.shape[1], input_img.shape[2])})
    new_model = Model(**configs)
    new_model.set_weights(_MODEL.get_weights())
    return new_model