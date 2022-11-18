import numpy as np
import os
import sys
import ntpath
import time
from . import util, html_util
from subprocess import Popen, PIPE
from PIL import Image
import json

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html_util.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for visual_group in visuals:
        for label, im_data in visual_group.items():
            im = util.tensor2im(im_data)
            image_name = "%s_%s.png" % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(im, save_path, aspect_ratio=aspect_ratio)
            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer:
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.output_display_id
        self.use_html = opt.isTrain and not opt.output_no_html
        self.win_size = opt.output_display_winsize
        self.name = opt.name
        self.port = opt.output_display_port
        self.saved = False
        self.metrics_dict = {}
        if (
            self.display_id > 0
        ):  # connect to a visdom server given <display_port> and <display_server>
            import visdom

            self.ncols = opt.output_display_ncols
            self.vis = visdom.Visdom(
                server=opt.output_display_server,
                port=opt.output_display_port,
                env=opt.output_display_env,
            )
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if (
            self.use_html
        ):  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, "web")
            self.img_dir = os.path.join(self.web_dir, "images")
            self.losses_path = os.path.join(self.web_dir, "losses.json")
            print("create web directory %s..." % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, "loss_log.txt")
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(
                "================ Training Loss (%s) ================\n" % now
            )

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port >"""
        cmd = sys.executable + " -m visdom.server -p %d &>/dev/null &" % self.port
        print("\n\nCould not connect to Visdom server. \n Trying to start a server....")
        print("Command: %s" % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result, params=[]):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:  # show all the images in one visdom panel
                max_ncol = 0
                for temp in visuals:
                    if max_ncol < len(temp):
                        max_ncol = len(temp)
                ncols = min(ncols, max_ncol)
                h, w = next(iter(visuals[0].values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (
                    w,
                    h,
                )  # create a table css
                # create a table of images.
                title = self.name
                label_html = ""
                label_html_row = ""
                param_html = ""
                param_html_row = ""
                images = []
                idx = 0
                for param in params.items():
                    param_html_row += "<td>%s</td>" % param[0]
                    param_html_row += "<td>%s</td>" % param[1]
                    param_html += "<tr>%s</tr>" % param_html_row
                    param_html_row = ""

                for visual_group in visuals:
                    label_html_row = ""
                    for label, image in visual_group.items():
                        image_numpy = util.tensor2im(image)
                        label_html_row += "<td>%s</td>" % label
                        images.append(image_numpy.transpose([2, 0, 1]))
                        idx += 1
                        if idx % ncols == 0:
                            label_html += "<tr>%s</tr>" % label_html_row
                            label_html_row = ""
                    white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                    while idx % ncols != 0:
                        images.append(white_image)
                        label_html_row += "<td></td>"
                        idx += 1
                    if label_html_row != "":
                        label_html += "<tr>%s</tr>" % label_html_row
                try:
                    self.vis.images(
                        images,
                        nrow=ncols,
                        win=self.display_id + 1,
                        padding=2,
                        opts=dict(title=title + " images"),
                    )
                    label_html = "<table>%s</table>" % label_html
                    param_html = "<table>%s</table>" % param_html
                    self.vis.text(
                        table_css + label_html,
                        win=self.display_id + 2,
                        opts=dict(title=title + " labels"),
                    )
                    self.vis.text(
                        table_css + param_html,
                        win=self.display_id + 3,
                        opts=dict(title=title + " params"),
                    )

                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:  # show each image in a separate visdom panel;
                idx = 1
                try:
                    for visual_group in visuals:
                        for label, image in visual_group.items():
                            image_numpy = util.tensor2im(image)
                            self.vis.image(
                                image_numpy.transpose([2, 0, 1]),
                                opts=dict(title=label),
                                win=self.display_id + idx,
                            )
                            idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (
            save_result or not self.saved
        ):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for visual_group in visuals:
                for label, image in visual_group.items():
                    image_numpy = util.tensor2im(image)
                    img_path = os.path.join(
                        self.img_dir, "epoch%.3d_%s.png" % (epoch, label)
                    )
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html_util.HTML(
                self.web_dir, "Experiment name = %s" % self.name, refresh=0
            )
            for n in range(epoch, 0, -1):
                webpage.add_header("epoch [%d]" % n)
                ims, txts, links = [], [], []

                for visual_group in visuals:
                    for label, image_numpy in visual_group.items():
                        image_numpy = util.tensor2im(image)
                        img_path = "epoch%.3d_%s.png" % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

        # Save latest images

        for visual_group in visuals:
            for label, image in visual_group.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, "latest_%s.png" % label)
                util.save_image(image_numpy, img_path)

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, "plot_data"):
            self.plot_data = {"X": [], "Y": [], "legend": list(losses.keys())}
        self.plot_data["X"].append(epoch + counter_ratio)
        if len(self.plot_data["legend"]) == 1:
            self.plot_data["Y"].append(losses[self.plot_data["legend"][0]])
            X = np.array(self.plot_data["X"])
            Y = np.array(self.plot_data["Y"])
        else:
            self.plot_data["Y"].append([losses[k] for k in self.plot_data["legend"]])
            X = np.stack(
                [np.array(self.plot_data["X"])] * len(self.plot_data["legend"]), 1
            )
            Y = np.array(self.plot_data["Y"])
        try:
            self.vis.line(
                Y,
                X,
                opts={
                    "title": self.name + " loss over time",
                    "legend": self.plot_data["legend"],
                    "xlabel": "epoch",
                    "ylabel": "loss",
                },
                win=self.display_id,
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()

        with open(self.losses_path, "w") as fp:
            json.dump(self.plot_data, fp)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data_mini_batch):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = (
            "(epoch: %d, iters: %d, time comput per image: %.3f, time data mini batch: %.3f) "
            % (epoch, iters, t_comp, t_data_mini_batch)
        )
        for k, v in losses.items():
            message += "%s: %.6f " % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write("%s\n" % message)  # save the message

    def display_img(self, img_path):
        im = Image.open(img_path)
        im = np.array(im)
        im = np.transpose(im, (2, 0, 1))
        img_name = img_path.split("/")[-1].split(".")[0]
        self.vis.image(im, opts=dict(title=self.name + " " + img_name))

    def plot_metrics_dict(
        self, name, epoch, counter_ratio, metrics, title, ylabel, win_id
    ):
        """Update a dict of metrics: labels and values and display it on visdom display

        Parameters:
            name (str)            -- identifier of the plot
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            metrics (OrderedDict) -- metrics stored in format (name, float)
            title (str)           -- Plot title
            ylabel (str)          -- y label
            window_id (int)       -- Visdom window id
        """
        if name not in self.metrics_dict:
            self.metrics_dict[name] = {"X": [], "Y": [], "legend": list(metrics.keys())}
        plot_metrics = self.metrics_dict[name]
        plot_metrics["X"].append(epoch + counter_ratio)
        plot_metrics["Y"].append([metrics[k] for k in plot_metrics["legend"]])
        X = np.stack([np.array(plot_metrics["X"])] * len(plot_metrics["legend"]), 1)
        Y = np.array(plot_metrics["Y"])
        try:
            # Resize needed due to a bug in visdom 0.1.8.9
            if Y.shape[1] == 1:
                X = X.reshape(X.shape[:1])
                Y = Y.reshape(Y.shape[:1])

            self.vis.line(
                Y,
                X,
                opts={
                    "title": self.name + " " + title,
                    "legend": plot_metrics["legend"],
                    "xlabel": "epoch",
                    "ylabel": ylabel,
                },
                win=self.display_id + win_id,
            )
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_current_fid(self, epoch, counter_ratio, fids):
        """display the current fid values on visdom display: dictionary of fid labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            fids (OrderedDict)  -- training fid values stored in the format of (name, float) pairs
        """
        self.plot_metrics_dict(
            "fid",
            epoch,
            counter_ratio,
            fids,
            title="fid over time",
            ylabel="fid",
            win_id=4,
        )

    def plot_current_D_accuracies(self, epoch, counter_ratio, accuracies):
        """display the current fid values on visdom display: dictionary of fid labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            accuracies (OrderedDict)  -- accuracy values stored in the format of (name, float) pairs
        """
        self.plot_metrics_dict(
            "D_accuracy",
            epoch,
            counter_ratio,
            accuracies,
            title="accuracy over time",
            ylabel="accuracy",
            win_id=5,
        )

    def plot_current_APA_prob(self, epoch, counter_ratio, p):
        self.plot_metrics_dict(
            "APA_prob",
            epoch,
            counter_ratio,
            p,
            title="APA params over time",
            ylabel="prob APA",
            win_id=6,
        )

    def plot_current_miou(self, epoch, counter_ratio, miou):
        """display the current fid values on visdom display: dictionary of fid labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            miouf_s (OrderedDict)  -- training miou_f_s values stored in the format of (name, float) pairs
        """
        self.plot_metrics_dict(
            "miou",
            epoch,
            counter_ratio,
            miou,
            title="miou over time",
            ylabel="miou",
            win_id=7,
        )
