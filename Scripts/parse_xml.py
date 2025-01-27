import xml.etree.ElementTree as ET

tree = ET.parse("annotations.xml")
images = tree.findall("image")

with open("labels_full.csv", "w+", encoding='utf-8') as f:
    for i in images:
        name = i.attrib['name']
        for p in i:
            if p.attrib['label'] != 'Display':
                continue
            text = p[0].text
            if text != "10000":
                f.write(f"{name}, {text}\n")
