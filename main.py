from PIL import Image
from random import sample

import xml.etree.cElementTree as ET
import ray, psutil
import shutil
import glob
import json
import os, sys



class ProgressLoader:
    def __init__(self):
        self.progress = 0
        self.length = 1
        self.blockCount = 20

        self.clear()
        self.printWhole("Waiting for process. Please wait a moment.")

    def printIncrement(self, i, subtext="", speed=7):
        if i % int(self.length / speed) == 0:
            self.printWhole(subtext)

    def printWhole(self, subtext=""):
        self.printProgress(self.progress, self.length, subtext)

    def printProgress(self, progress, length, text=""):
        progressPercentage = (progress / length * 100.0)
        count = int(progressPercentage / 100.0 * self.blockCount)
        
        progressBar = "█" * count + "░" * (self.blockCount - count)
        text = str(text)
        print(" [- " + progressBar + " {:.1f}% -]".format(progressPercentage) + " " + text, end='\r')

    def updateLength(self, length, increment=False):
        if increment: self.length += 1
        else: self.length = length

    def updateProgress(self, amount=1, reset=False):
        if reset: self.progress = 0
        else:
            self.progress += amount

    def clear(self):
        os.system('cls')
        os.system('clear')


class DataFormat:
    def __init__(self, resize):
        self.pLoader = ProgressLoader()
        self.resize = resize
        self.inputPath = "./xml/"
        self.directoryList, _ = self.getDirList(self.inputPath)

        num_cpus = psutil.cpu_count(logical=False)
        ray.init(num_cpus=num_cpus, log_to_driver=True)

    def getDirList(self, searchDir):
        for base, dirs, files in os.walk(searchDir):
            directoryList = []
            fileList = []
            for directory in dirs:
                directoryList.append(directory)
            for file in files:
                fileList.append(file)
            return directoryList, fileList

    @ray.remote
    def resizeImages(self, width, inputDir, outputDir):
        _, fileList = self.getDirList(inputDir)
        for fileName in fileList[1:]:
            if "xml" in fileName: continue
            self.resizeImage(width, inputDir, outputDir, fileName)

    def resizeImage(self, width, inputDir, outputDir, imgName):
        image = Image.open(inputDir + "/" + imgName)
        newImage = image.resize((width, width), Image.NEAREST)
        newImage.save(outputDir + "/" + imgName)

    def getRoot(self, dirName):
        xmlDir = glob.glob(self.inputPath + dirName + "/*.xml")[0]
        tree = ET.parse(xmlDir)
        root = tree.getroot()
        return root

    def getCategories(self, root):
        categoryList = []
        categoryDict = {}

        labels = root[1][0][12].findall('label')
        for i, label in enumerate(labels):
            category = {"supercategory": "none", "id": i+1, "name": label[0].text}
            categoryList.append(category)
            categoryDict[label[0].text] = i+1
        return (categoryList, categoryDict)

    def clearDir(self, dirName):
        shutil.rmtree(dirName)
        os.mkdir(dirName)
            

    def toIterator(self, objIds):
        while objIds:
            done, objIds = ray.wait(objIds)
            yield ray.get(done[0])

    def executeRay(self, tasks=[], subtext=""):
        self.pLoader.clear()
        self.pLoader.updateLength(len(tasks))
        self.pLoader.updateProgress(reset=True)
        self.pLoader.printWhole(subtext)
        for i, _ in enumerate(self.toIterator(tasks)):
            self.pLoader.updateProgress()
            self.pLoader.printWhole(subtext)
        self.pLoader.printWhole(subtext)



class YoloFormat(DataFormat):
    def __init__(self, resize, valid=20, test=5):
        super().__init__(resize)
        self.categoryDict = {}
        self.valid = valid
        self.test = test
        self.outputPath = "./yolo/"
        self.trainImgPath = ""
        self.trainLabelPath = ""
        self.validImgPath = ""
        self.validLabelPath = ""
        self.testImgPath = ""
        self.testLabelPath = ""
        self.createYolo()        

    def createYolo(self):
        self.clearDir(self.outputPath)
        self.initDir()
        self.initCategory()
        self.createLabels()
        self.createYaml()
        self.resizeImages()
        self.shuffleData(self.outputPath + self.trainImgPath)

    def initDir(self):
        self.trainImgPath = "train/images/"
        self.trainLabelPath = "train/labels/"
        self.validImgPath = "valid/images/"
        self.validLabelPath = "valid/labels/"
        self.testImgPath = "test/images/"
        self.testLabelPath = "test/labels/"

        outputList = ["train", "valid", "test"]
        for dirName in outputList:
            outputDir = self.outputPath + dirName
            os.mkdir(outputDir)
            os.mkdir(outputDir + "/images")
            os.mkdir(outputDir + "/labels")

    def initCategory(self):
        for dirName in self.directoryList:
            root = self.getRoot(dirName)
            _, subDict = self.getCategories(root)
            for key, value in subDict.items():
                self.categoryDict[key] = None
        for i, pair in enumerate(self.categoryDict.items()):
            self.categoryDict[pair[0]] = i
        
    def createLabels(self):
        tasksPreLaunch = []
        for dirName in self.directoryList:
            root = self.getRoot(dirName)
            tasksPreLaunch.append(self.createLabel.remote(self, root))
        self.executeRay(tasksPreLaunch, "Creating Labels")

    def shuffleData(self, source):
        fileList = []
        count = 0
        for base, dirs, files in os.walk(source):
            for file in files:
                count += 1
                fileList.append(file)

        validCount = int(count / 100.0 * self.valid)
        testCount = int(count / 100.0 * self.test)
        sampleList = sample(fileList, validCount+testCount)
        validList = sampleList[:validCount]
        testList = sampleList[validCount:]

        progressText = "Shuffling valid, test data"

        self.pLoader.updateLength(len(validList) + len(testList))
        self.pLoader.updateProgress(reset=True)
        for i, imgName in enumerate(validList):
            shutil.move(self.outputPath+self.trainImgPath+imgName, self.outputPath+self.validImgPath)
            labelName = imgName[:-4] + ".txt"
            shutil.move(self.outputPath+self.trainLabelPath+labelName, self.outputPath+self.validLabelPath)
            self.pLoader.updateProgress()
            self.pLoader.printWhole(progressText)

        dest = self.outputPath + "test/"
        self.pLoader.updateLength(len(testList))
        self.pLoader.updateProgress(reset=True)
        for i ,imgName in enumerate(testList):
            shutil.move(self.outputPath+self.trainImgPath+imgName, self.outputPath+self.testImgPath)
            labelName = imgName[:-4] + ".txt"
            shutil.move(self.outputPath+self.trainLabelPath+labelName, self.outputPath+self.testLabelPath)
            self.pLoader.updateProgress()
            self.pLoader.printWhole(progressText)

    def createYaml(self):
        with open(self.outputPath + "data.yaml", "w", encoding="utf-8") as f:
            f.write("train: ../" + self.trainImgPath + "\n")
            f.write("val: ../" + self.validImgPath + "\n")
            f.write("test: ../" + self.testImgPath + "\n\n")
            f.write("nc: " + str(len(self.categoryDict)) + "\n")
            f.write("names: " + "['" + "','".join(self.categoryDict.keys()) + "']")
            f.close()

    def resizeImages(self):
        tasksPreLaunch = []
        resize = ray.put(self.resize)
        outputPath = ray.put(self.outputPath + self.trainImgPath)
        for dirName in self.directoryList:
            tasksPreLaunch.append(super().resizeImages.remote(
                    self, resize, self.inputPath + dirName, outputPath
                )
            )
        self.executeRay(tasksPreLaunch, "Resizing Images")

    @ray.remote
    def createLabel(self, root):
        imageTags = root.findall("image")
        for imageTag in imageTags:
            name = imageTag.attrib['name'][:-4]
            height = float(imageTag.attrib['height'])
            width = float(imageTag.attrib['width'])

            widthDiv, heightDiv = width/self.resize, height/self.resize
            with open(self.outputPath + self.trainLabelPath + name + ".txt", "w") as f:
                self.computeShapes(f, imageTag, widthDiv, heightDiv)

    def computeShapes(self, f, imageTag, widthDiv, heightDiv):
        for boxTag in imageTag.findall("box"):
            attr = boxTag.attrib
            label = attr["label"]
            xtl, ytl, xbr, ybr = float(attr['xtl']), float(attr['ytl']), \
                float(attr['xbr']), float(attr['ybr'])
            xtl, ytl, xbr, ybr = xtl/widthDiv, ytl/heightDiv, xbr/widthDiv, ybr/heightDiv

            category = self.categoryDict[label]
            xCenter = ((xtl+xbr) / 2.0) / self.resize
            yCenter = ((ytl+ybr) / 2.0) / self.resize
            boxWidth = abs(xtl-xbr) / self.resize
            boxHeight = abs(ytl-ybr) / self.resize

            outputLine = "{} {} {} {} {}".format(category, xCenter, yCenter, boxWidth, boxHeight)
            f.write(outputLine + "\n")



class YoloPolyFormat(YoloFormat):
    def __init__(self, resize, valid=20, test=5):
        self.polygonMaxCount = 5
        super().__init__(resize, valid, test)
        self.createImgPathFile()

    def countMaxPolygonCount(self, root):
        imageTags = root.findall("image")
        for imageTag in imageTags:
            for polygonTag in imageTag.findall("polygon"): 
                points = polygonTag.attrib["points"].split(";")
                count = len(points)
                if self.polygonMaxCount < count:
                    self.polygonMaxCount = count

    def computeShapes(self, f, imageTag, widthDiv, heightDiv):
        for polygonTag in imageTag.findall("polygon"):
            attr = polygonTag.attrib
            label = attr["label"]
            points = attr["points"].split(";")

            category = self.categoryDict[label]
            outputLine = str(category)
            excessCount = self.polygonMaxCount - len(points)
            excessPoints = [points[-1]] * excessCount
            for point in (points[:5] + excessPoints):
                x, y = point.split(",")
                x, y = float(x) / widthDiv, float(y) / heightDiv
                x, y = x / self.resize, y / self.resize
                outputLine += " " + "{:.6f}".format(x) + " " + "{:.6f}".format(y)
            f.write(outputLine + "\n")

    def initDir(self):
        self.trainImgPath = "images/train/"
        self.trainLabelPath = "labels/train/"
        self.validImgPath = "images/valid/"
        self.validLabelPath = "labels/valid/"
        self.testImgPath = "images/test/"
        self.testLabelPath = "labels/test/"

        outputList = ["images", "labels"]
        for dirName in outputList:
            outputDir = self.outputPath + dirName
            os.mkdir(outputDir)
            os.mkdir(outputDir + "/train")
            os.mkdir(outputDir + "/valid")
            os.mkdir(outputDir + "/test")

        # for dirName in self.directoryList:
        #     root = self.getRoot(dirName)
        #     self.countMaxPolygonCount(root)

    def createYaml(self):
        with open(self.outputPath + "data.yaml", "w", encoding="utf-8") as f:
            f.write("train: ../data/" + "train.txt" + "\n")
            f.write("val: ../data/" + "valid.txt" + "\n")
            f.write("test: ../data/" + "test.txt" + "\n\n")
            f.write("nc: " + str(len(self.categoryDict)) + "\n")
            f.write("names: " + "['" + "','".join(self.categoryDict.keys()) + "']")
            f.close()

    def createImgPathFile(self):
        sourceDict = {"train":self.trainImgPath, "valid":self.validImgPath, "test":self.testImgPath}

        for key, value in sourceDict.items():
            for base, dirs, files in os.walk(self.outputPath + value):
                files.sort()
                with open(self.outputPath + key + ".txt", "w") as f:
                    for file in files:
                        f.write("../data/" + value + file + "\n")



class CocoFormat(DataFormat):
    def __init__(self, resize):
        super().__init__(resize)
        self.ouputPath = "./coco/"

        self.createCoco()

    def createCoco(self):
        jsonTasks = []
        resizeTasks = []
        resize = ray.put(self.resize)
        for dirName in self.directoryList:
            inputDir = self.inputPath + dirName
            outputDir = self.ouputPath + dirName

            jsonTasks.append(self.createJson.remote(self, dirName, outputDir))
            resizeTasks.append(self.resizeImages.remote(self, resize, inputDir, outputDir))
        self.executeRay(jsonTasks, "Creating JSON files")
        self.executeRay(resizeTasks, "Resizing Images")

    @ray.remote
    def createJson(self, dirName, outputDir):
        root = self.getRoot(dirName)
        info = self.getInfo(root)
        licenses = self.getLicenses()
        categories, categoryDict = self.getCategories(root)
        images = self.getImages(root)
        annotations = self.getAnnotations(root, categoryDict)

        coco = {
            "info": info,
            "licenses": licenses,
            "images": images,
            "annotations": annotations,
            "categories": categories,
            "segment_info": []
        }

        os.mkdir(outputDir)
        with open(outputDir + "/" + dirName + ".json", "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False, indent=4)

    def getInfo(self, root):
        task = root[1][0]

        description = ""
        url = task[13][0][3].text
        version = root[0].text
        year = 2019
        contributor = ""
        date_created = task[7].text

        return {
            "description": description, "url": url, "version": version, "year": year,
            "contributor": contributor, "date_created": date_created
        }

    def getLicenses(self):
        return {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        }

    def getImages(self, root):
        imageList = []

        for imageTag in root.findall("image"):
            attr = imageTag.attrib
            id = int(attr['id'])
            name = attr['name']
            height = self.resize
            width = self.resize

            image = {
                "license": 1,
                "file_name": name,
                "coco_url": "",
                "height": height,
                "width": width,
                "date_captured": "",
                "flickr_url": "",
                "id": id
            }
            imageList.append(image)

        return imageList

    def getAnnotations(self, root, categoryDict):
        annotationList = []

        for imageTag in root.findall("image"):
            imgAttr = imageTag.attrib
            for i, boxTag in enumerate(imageTag):
                attr = boxTag.attrib
                segmentation = []

                imgWidth, imgHeight = float(imgAttr['width']), float(imgAttr['height'])
                widthDiv, heightDiv = imgWidth / self.resize, imgHeight / self.resize
                xtl, ytl, xbr, ybr = float(attr['xtl']), float(attr['ytl']), \
                    float(attr['xbr']), float(attr['ybr'])
                xtl, ytl, xbr, ybr = xtl/widthDiv, ytl/heightDiv, xbr/widthDiv, ybr/heightDiv

                area = 0
                iscrowd = 0
                image_id = int(imgAttr['id'])
                bbox = [xtl, ytl, abs(xtl-xbr), abs(ytl-ybr)]
                category_id = categoryDict[attr['label']]
                id = int("{}{}{}".format(image_id, 0, i))

                annotation = {
                    "segmentation": segmentation,
                    "area": area,
                    "iscrowd": iscrowd,
                    "image_id": image_id,
                    "bbox": bbox,
                    "category_id": category_id,
                    "id": id
                }
                annotationList.append(annotation)
        
        return annotationList



if __name__ == '__main__':
    yolo = YoloPolyFormat(640)