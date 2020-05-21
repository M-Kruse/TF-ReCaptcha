from time import sleep
import requests
from selenium import webdriver
from PIL import Image
import random
import string
import cv2

class RecaptchaElement(object):
    """docstring for RecaptchaElement"""
    def __init__(self, element, row, col, img, type):
        super(RecaptchaElement, self).__init__()
        self.row = row
        self.column = col
        self.element = element
        self.img = img
        self.type = type
                
    def click(self):
        self.element.click()

    def render_img(self):
        img = Image.fromarray(self.img)
        img.show()

class TFRecaptcha(object):
    """docstring for TFRecaptcha"""
    def __init__(self):
        super(TFRecaptcha, self).__init__()
        self.test_url = "https://patrickhlauke.github.io/recaptcha/"
        self.recaptcha_types = [
            'fire hydrant',
            'fire hydrants',
            'bicycle',
            'bicycles',
            'traffic light',
            'traffic lights',
            'bus',            
            'buses',         
            'taxis',
            'cars',
            'motorcycle',
            'motorcycles',
            'crosswalk'
            'crosswalks'
        ]
        """
        The individual recaptcha image elements will be saved in
        the same row/column order that the HTML elements are in
        """
        self.elements = []
        self.recaptchas = []
        #Bool for determining harvesting or solving mode
        self.harvest_mode = False

    """ Initalize the Firefox browser """
    def init_browser(self):
        self.browser = webdriver.Firefox()
        if self.browser:
            return True
        else:
            print("[ERROR] TFRecaptcha::init_browser(): Failed to create Firefox webdriver instance.")
            return False

    def detect_recaptcha_type(self, target_types=None):
        """
        This tries to get the text from the ReCaptcha element to determine what class of object is requested
        """
        try:
            raw = self.browser.page_source
            for captcha_type in self.recaptcha_types:
                if captcha_type in raw:
                    if target_types is None:
                        return True, captcha_type
                    else:
                        for target in target_types:
                            if target in raw:
                                return True, captcha_type
                        else:
                            return False, captcha_type
            return False, None
        except Exception as e:
            print("[ERROR] TFRecaptcha::detect_recaptcha_type()): {0}".format(e))
            return False, None
        return False, None

    def is_3x3_image_grid(self):
        """ 
        The recaptchas seem to either be 3x3 or 4x4 puzzles.
        The 3x3 is a single image but cut up into 3x3 grid
        The 4x4 is a single image with something to make it more like an overlay grid.
        """
        return bool(self.browser.find_elements_by_class_name("rc-image-tile-33"))

    def download_url(self, url, filename):
        """
        Used for downloading the images from URL
        """
        data = requests.get(url).content
        with open(filename, 'wb') as handler:
            handler.write(data)

    def download_recaptcha_img(self, type):
        """
        Used to detect what type of image and handle downloading it for processing
        """
        rand = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(6)])
        try:
            img_src = self.browser.find_element_by_class_name("rc-image-tile-44").get_attribute("src")
        except:
            img_src = self.browser.find_element_by_class_name("rc-image-tile-33").get_attribute("src")
        if img_src:
            if type:

                file_path = "imgs/{0}/{1}.png".format(type, rand)
            else:
                file_path = "imgs/unknown/{1}.png".format(type, rand)
            self.download_url(img_src, file_path)
            return file_path
        else:
            print("[ERROR] TFRecaptcha::download_recaptcha_img(): Failed to download recaptcha image.")

    def get_recaptcha_elements(self):
        """
        Gets a list of the individual recaptcha elements
        """
        self.elements = self.browser.find_elements_by_class_name("rc-image-tile-wrapper")
        
    def open_recaptcha(self):
        """
        This will handle the funny process of getting to the frame for interacting with the recaptcha
        This took hours to figure out
        """
        self.browser.get(self.test_url)
        sleep(2)
        self.browser.find_elements_by_xpath("//iframe[contains(@src, 'google')]")[0].click()
        sleep(2)
        #Sauce
        recaptcha_iframe = self.browser.find_element_by_xpath("//iframe[contains(@title, 'recaptcha challenge')]")
        self.browser.switch_to.frame(recaptcha_iframe)
        if not self.is_3x3_image_grid():
            self.attempt_puzzle_type_bypass()

    def detect_denial_of_service():
        if self.browser.find_element_by_class_name("rc-doscaptcha-header"):
            return True
        else:
            return False

    def close_browser(self):
        self.browser.close()

    def click_solve_or_skip(self):
        self.browser.find_element_by_class_name("rc-button-default").click()

    def attempt_puzzle_type_bypass(self, retries=24):
        is_3x3 = self.is_3x3_image_grid()
        for i in range(1, retries):
            self.click_solve_or_skip()
            sleep(1)
            new_type = self.is_3x3_image_grid()

    def attempt_object_type_bypass(self, retries=24):
        ret, type = self.detect_recaptcha_type()
        if ret:
            for i in range(1, retries):
                self.click_solve_or_skip()
                sleep(1)
                new_type = self.detect_recaptcha_type()
                if type == new_type:
                    continue
                else:
                    print("[INFO] Detected new puzzle object type: {0}".format(new_type))
                    break

    def generate_recaptcha_classes(self):
        """
        This will try to turn the ReCaptcha elements into seperate RecaptchaElement classes
        """
        self.get_recaptcha_elements()
        ret, type = self.detect_recaptcha_type()
        if self.is_3x3_image_grid():
            grid_size = 3
        else:
            grid_size = 4
        saved_img = self.download_recaptcha_img(type)
        if self.harvest_mode:
            im =  cv2.imread(saved_img)
            imgheight=im.shape[0]
            imgwidth=im.shape[1]
            y1 = 0
            M = imgheight//grid_size
            N = imgwidth//grid_size
            index = -1
            for x in range(0, imgheight, N):
                for y in range(0, imgwidth, M):
                    index += 1
                    y1 = y + M
                    x1 = x + N
                    tile = im[x:x+N,y:y+M,]
                    cv2.rectangle(im, (x, y), (x1, y1), (0, 0, 0))
                    if self.harvest_mode:
                        rand = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(6)])
                        file_path = "imgs/{0}/{1}.png".format(type, rand)
                        cv2.imwrite(file_path, tile)
                        print("Saved Img: {0}".format(file_path))
                    r = RecaptchaElement(self.elements[index], x, y, tile, type)
                    self.recaptchas.append(r) 

if __name__ == "__main__":
    TFR = TFRecaptcha()
    while True:
        TFR.init_browser()
        #Harvest mode just skips the solving and only saves the puzzle images
        TFR.harvest_mode = True
        TFR.open_recaptcha()
        TFR.generate_recaptcha_classes()
        TFR.close_browser()
    # TFR.generate_recaptcha_classes()
    # for i in TFR.recaptchas:
    #     i.render_img()
    #     i.click()

