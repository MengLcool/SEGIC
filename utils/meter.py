import torch
from . import misc

class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, class_ids, logger):
        self.logger = logger
        self.class_ids_interest = class_ids
        self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()

        self.nclass = 2000

        self.intersection_buf = torch.zeros([self.nclass]).float().cuda()
        self.union_buf = torch.zeros([self.nclass]).float().cuda()
        self.ones = torch.ones_like(self.union_buf)
        self.loss_buf = []
        self.valid_ids = torch.zeros([self.nclass]).float().cuda()

    def update(self, inter_b, union_b, class_id, loss):
        if isinstance(class_id, torch.Tensor):
            self.valid_ids[class_id.unique()] +=1
        elif isinstance(class_id, int):
            self.valid_ids[class_id] +=1

        self.intersection_buf[class_id] += inter_b
        self.union_buf[class_id] += union_b
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)

    def compute_iou(self):
        intersection_buf = torch.stack(misc.all_gather(self.intersection_buf.cpu())).sum(0).cuda()
        union_buf = torch.stack(misc.all_gather(self.union_buf.cpu())).sum(0).cuda()

        iou = intersection_buf.float() / union_buf.float().clip(min=1)
        # iou = iou[self.class_ids_interest]
        valid_ids = torch.stack(misc.all_gather(self.valid_ids.cpu())).sum(0).cuda()
        iou = iou[valid_ids > 0]
        print('shsape', iou.shape)
        miou = iou.mean() * 100

        return miou, None

    def write_result(self, split=0):
        iou, fb_iou = self.compute_iou()

        loss_buf = torch.stack(self.loss_buf)
        msg = '\n*** %s ' % split
        msg += 'Avg L: %6.5f  ' % loss_buf.mean()
        msg += 'mIoU: %5.2f   ' % iou

        msg += '***\n'
        self.logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
            iou, fb_iou = self.compute_iou()
            if epoch != -1:
                loss_buf = torch.stack(self.loss_buf)
                msg += 'L: %6.5f  ' % loss_buf[-1]
                msg += 'Avg L: %6.5f  ' % loss_buf.mean()
            msg += 'mIoU: %5.2f  |  ' % iou
            msg += 'FB-IoU: %5.2f' % fb_iou
            self.logger.info(msg)