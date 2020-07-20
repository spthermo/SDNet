import visdom
import numpy
import torch
import os

class NullVisualizer(object):
    def __init__(self):
        self.name = __name__

    def append_loss(self, epoch, global_iteration, loss, mode='train'):
        pass

    def show_images(self, images, title):
        pass

class VisdomVisualizer(object):
    def __init__(self, name, server="http://localhost", count=1):
        self.visualizer = visdom.Visdom(server=server, port=8097, env=name,\
            use_incoming_socket=False)
        self.name = name
        self.first_train_value = True
        self.first_test_value = True
        self.count = count
        self.plots = {}
        
    def append_loss(self, epoch, global_iteration, loss, loss_name="total", mode='train'):
        plot_name = loss_name + '_train_loss' if mode == 'train' else 'test_loss'
        opts = (
            {
                'title': plot_name,
                #'legend': mode,
                'xlabel': 'iterations',
                'ylabel': loss_name
            })
        loss_value = loss #float(loss.detach().cpu().numpy())
        if loss_name not in self.plots:
            self.plots[loss_name] = self.visualizer.line(X=numpy.array([global_iteration]), Y=numpy.array([loss_value]), opts=opts)
        else:
            self.visualizer.line(X=numpy.array([global_iteration]), Y=numpy.array([loss_value]), win=self.plots[loss_name], name=mode, update = 'append')


    def show_seg_map(self, in_map, title, iter=0):
        # maps = in_map.squeeze(0).detach()
        # h, w = maps.size()
        b, c, h, w = in_map.size()
        maps_cpu = in_map.detach().cpu()[:self.count, :, :, :]  
        maps_cpu = in_map.cpu()[:, :]
        for i in range(self.count):
            for j in range(4):
                opts = (
                {
                    'title': title + '_' + str(j), 'colormap': 'Viridis'
                })
                heatmap = maps_cpu[i, j, :, :].squeeze(0)
                self.visualizer.heatmap(heatmap.squeeze(0),\
                    opts=opts, win=self.name + title + "_window_" + str(j))

    def show_anatomical_factors(self, in_map, title, iter=0):
        # maps = in_map.squeeze(0).detach()
        # h, w = maps.size()
        b, c, h, w = in_map.size()
        maps_cpu = in_map.detach().cpu()[:self.count, :, :, :]  
        maps_cpu = in_map.cpu()[:, :]
        for i in range(self.count):
            for j in range(8):
                opts = (
                {
                    'title': title + '_' + str(j), 'colormap': 'Viridis'
                })
                heatmap = maps_cpu[i, j, :, :].squeeze(0)
                self.visualizer.heatmap(heatmap.squeeze(0),\
                    opts=opts, win=self.name + title + "_window_" + str(j))

    def show_map(self, maps, title):
        b, c, h, w = maps.size()
        maps_cpu = maps.detach().cpu()[:self.count, :, :, :]
        for i in range(self.count):
            opts = (
            {
                'title': title + str(i), 'colormap': 'Viridis'
            })
            heatmap = maps_cpu[i, :, :, :].squeeze(0)
            self.visualizer.heatmap(heatmap.squeeze(0),\
                opts=opts, win=self.name + title + "_window_" + str(i))

