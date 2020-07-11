def properties(hcam):
    print("Supported properties:")
    props = hcam.getProperties()
    for i, id_name in enumerate(sorted(props.keys())):
        [p_value, p_type] = hcam.getPropertyValue(id_name)
        p_rw = hcam.getPropertyRW(id_name)
        read_write = ""
        if (p_rw[0]):
            read_write += "read"
        if (p_rw[1]):
            read_write += ", write"
        print("  ", i, ")", id_name, " = ", p_value, " type is:", p_type, ",", read_write)
        text_values = hcam.getPropertyText(id_name)
        if (len(text_values) > 0):
            print("          option / value")
            for key in sorted(text_values, key = text_values.get):
                print("         ", key, "/", text_values[key])
