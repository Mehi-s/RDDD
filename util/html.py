import dominate
from dominate.tags import *
import os


class HTML:
  """A helper class for generating HTML pages.

  This class simplifies the process of creating an HTML page with a title,
  headers, and tables of images with captions.

  Args:
    web_dir (str): The directory where the HTML file and associated images will be saved.
    title (str): The title of the HTML page.
    reflesh (int, optional): If greater than 0, adds a meta tag to auto-refresh
                             the page every `reflesh` seconds. Defaults to 0.
  """
    def __init__(self, web_dir, title, reflesh=0):
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        # print(self.img_dir)

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

    def get_image_dir(self):
      """Returns the directory where images are stored.

      Returns:
        str: The path to the image directory.
      """
        return self.img_dir

    def add_header(self, str_content):
      """Adds a level 3 header to the HTML page.

      Args:
        str_content (str): The text content of the header.
      """
        with self.doc:
            h3(str_content)

    def add_table(self, border=1):
      """Adds a table to the HTML page.

      Args:
        border (int, optional): The border width of the table. Defaults to 1.
      """
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, height=400):
      """Adds a row of images with captions and links to the current table.

      A new table is created if one doesn't exist.

      Args:
        ims (list of str): A list of image filenames (relative to the image directory).
        txts (list of str): A list of text captions for each image.
        links (list of str): A list of link filenames (relative to the image directory) for each image.
        height (int, optional): The display height of the images in pixels. Defaults to 400.
      """
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="height:%dpx" % height, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
      """Saves the generated HTML content to an 'index.html' file in the web directory."""
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims = []
    txts = []
    links = []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()
