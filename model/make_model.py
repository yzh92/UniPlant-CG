import torch
import torch.nn as nn
# from .backbones.resnet import ResNet, Bottleneck,BasicBlock
from .backbones.resnet_attention import ResNet,BasicBlock,Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
import torchvision
import math
import torch.nn.functional as F
import torchvision.models as models
def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')

        elif model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, block=BasicBlock, layers=[2,2,2,2])
            print('using resnet18 as a backbone')

        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, block=BasicBlock, layers=[3,4,6,3])
            print('using resnet34 as a backbone')

        elif model_name == 'denseNet121':
            self.in_planes = 1024
            self.base = models.densenet121(pretrained=(pretrain_choice == 'imagenet')).features
            print('using DenseNet121 as a backbone')

        elif model_name == 'shuffleNet':
            self.in_planes = 1024
            self.base = nn.Sequential(
                *list(models.shufflenet_v2_x1_0(pretrained=(pretrain_choice == 'imagenet')).children())[:-1]
            )
            print('using ShuffleNetV2 x1.0 as a backbone')

        elif model_name == 'efficientNet':
            self.in_planes = 1280
            self.base = models.efficientnet_b0(pretrained=(pretrain_choice == 'imagenet')).features
            print('using EfficientNet-B0 as a backbone')
        
        elif model_name == 'mobileNet':
            self.in_planes = 1280
            self.base = models.mobilenet_v2(pretrained=(pretrain_choice == 'imagenet')).features
            print('using MobileNetV2 as a backbone')

        else:
            print(f'Unsupported backbone! Got {model_name}')

        if pretrain_choice == 'imagenet' and 'resnet' in model_name:
            self.base.load_param(model_path)
            print(f'Loading pretrained ImageNet model from {model_path}')
        elif pretrain_choice == 'imagenet':
            print(f'Using torchvision pretrained weights for {model_name}')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    # def load_param(self, trained_path):
    #     param_dict = torch.load(trained_path)
    #     if 'state_dict' in param_dict:
    #         param_dict = param_dict['state_dict']
    #     for i in param_dict:
    #         self.state_dict()[i].copy_(param_dict[i])
    #     print('Loading pretrained model from {}'.format(trained_path))
    
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        
        model_dict = self.state_dict()
        loaded_keys = []
        skipped_keys = []

        for name in param_dict:
            if name not in model_dict:
                skipped_keys.append(name)
                continue
            if model_dict[name].shape != param_dict[name].shape:
                skipped_keys.append(name)
                continue
            model_dict[name].copy_(param_dict[name])
            loaded_keys.append(name)

        print(f'✅ Loaded pretrained model from: {trained_path}')
        print(f'✅ Loaded {len(loaded_keys)} layers.')
        if skipped_keys:
            print(f'⚠️ Skipped {len(skipped_keys)} layers (e.g. classification layer or shape mismatch):')
            print(skipped_keys[:5])  # 只打印前5个以防太长

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True, embed_dim=768):
        super(ResNetBackbone, self).__init__()
        # Load pre-trained ResNet model
        self.resnet = torchvision.models.resnet50(pretrained = pretrained)
        # Remove the final fully connected layer and adjust the output feature dimension
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.fc = nn.Linear(2048, embed_dim)  # Adjusting the feature dimension to match `embed_dim`

    def forward(self, x):
        x = self.resnet(x)
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = self.fc(x)  # Adjust feature dimension
        return x

class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        # self.base = ResNetBackbone(embed_dim=self.in_planes)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))
        elif pretrain_choice == 'dino':
            self.base.load_param(model_path,flag=1)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))


        block = self.base.blocks[-1]
        # block = self.base[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)


    def load_param(self, model_path, dino_finetune_flag=False):
        print(f"Loading weights from: {model_path}")
        
        if dino_finetune_flag:
            param_dict1 = torch.load(model_path, map_location='cpu')
            param_dict1 = param_dict1['teacher']
            param_dict = {k.replace('backbone.', 'base.'): v for k, v in param_dict1.items() if k.startswith('backbone.')}
        else:
            param_dict = torch.load(model_path, map_location='cpu')

        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']

        model_state = self.state_dict()

        # 新增调试输出：查看权重文件中的键
        print(f"[DEBUG] Keys in loaded param_dict (first 10): {list(param_dict.keys())[:10]}")
        print(f"[DEBUG] Keys in model_state (first 10): {list(model_state.keys())[:10]}")
        
        loaded_keys = []
        skipped_keys = []
        mismatched_keys = []

        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                skipped_keys.append(k)
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.base.pos_embed.shape:
                if 'distilled' in model_path:
                    print('[Warn] Using distilled model, adjusting pos_embed tokens...')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.base.pos_embed, self.base.patch_embed.num_y, self.base.patch_embed.num_x)

            if k in model_state:
                try:
                    model_state[k].copy_(v)
                    loaded_keys.append(k)
                except Exception as e:
                    mismatched_keys.append((k, v.shape, model_state[k].shape))
            else:
                skipped_keys.append(k)

        print(f"[INFO] Loaded parameters: {len(loaded_keys)}")
        print(f"[INFO] Skipped parameters (not in model): {len(skipped_keys)}")
        if skipped_keys:
            print("  First 10 skipped keys:", skipped_keys[:10])
        if mismatched_keys:
            print(f"[WARN] Mismatched parameters: {len(mismatched_keys)}")
            for k, pre_shape, model_shape in mismatched_keys[:5]:  # 只显示前5个
                print(f"  - {k}: pretrained {pre_shape} vs model {model_shape}")

        # 新增：检查是否有共同的前缀需要去除
        if len(loaded_keys) == 0 and len(param_dict) > 0:
            print("\n[DEBUG] Trying to find common prefix pattern...")
            # 尝试去除可能的前缀
            common_prefixes = ['module.', 'backbone.', 'encoder.', 'model.']
            for prefix in common_prefixes:
                modified_dict = {}
                for k, v in param_dict.items():
                    if k.startswith(prefix):
                        new_key = k.replace(prefix, '')
                        modified_dict[new_key] = v
                
                # 检查修改后的匹配情况
                matched_count = 0
                for k in modified_dict:
                    if k in model_state:
                        matched_count += 1
                
                if matched_count > 0:
                    print(f"Found pattern: removing '{prefix}' gives {matched_count} matches")
                    # 重新加载参数
                    for k, v in modified_dict.items():
                        if k in model_state and k not in loaded_keys:
                            try:
                                model_state[k].copy_(v)
                                loaded_keys.append(k)
                            except Exception as e:
                                pass
                    
                    print(f"After pattern adjustment, loaded {len(loaded_keys)} parameters")
                    break

        return len(loaded_keys) > 0  # 返回是否成功加载了参数


    # # 0904
    # def load_param(self, model_path, dino_finetune_flag=False):
    #     print(f"Loading weights from: {model_path}")
        
    #     if dino_finetune_flag:
    #         param_dict1 = torch.load(model_path, map_location='cpu')
    #         param_dict1 = param_dict1['teacher']
    #         param_dict = {k.replace('backbone.', ''): v for k, v in param_dict1.items() if k.startswith('backbone.')}
    #     else:
    #         param_dict = torch.load(model_path, map_location='cpu')

    #     if 'model' in param_dict:
    #         param_dict = param_dict['model']
    #     if 'state_dict' in param_dict:
    #         param_dict = param_dict['state_dict']

    #     model_state = self.state_dict()

    #     loaded_keys = []
    #     skipped_keys = []
    #     mismatched_keys = []

    #     for k, v in param_dict.items():
    #         if 'head' in k or 'dist' in k:
    #             skipped_keys.append(k)
    #             continue
    #         if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
    #             O, I, H, W = self.patch_embed.proj.weight.shape
    #             v = v.reshape(O, -1, H, W)
    #         elif k == 'pos_embed' and v.shape != self.base.pos_embed.shape:
    #             if 'distilled' in model_path:
    #                 print('[Warn] Using distilled model, adjusting pos_embed tokens...')
    #                 v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
    #             v = resize_pos_embed(v, self.base.pos_embed, self.base.patch_embed.num_y, self.base.patch_embed.num_x)

    #         if k in model_state:
    #             try:
    #                 model_state[k].copy_(v)
    #                 loaded_keys.append(k)
    #             except Exception as e:
    #                 mismatched_keys.append((k, v.shape, model_state[k].shape))
    #         else:
    #             skipped_keys.append(k)

    #     print(f"[INFO] Loaded parameters: {len(loaded_keys)}")
    #     print(f"[INFO] Skipped parameters (not in model): {len(skipped_keys)}")
    #     if skipped_keys:
    #         print("  ", skipped_keys)
    #     if mismatched_keys:
    #         print(f"[WARN] Mismatched parameters: {len(mismatched_keys)}")
    #         for k, pre_shape, model_shape in mismatched_keys:
    #             print(f"  - {k}: pretrained {pre_shape} vs model {model_shape}")

    def resize_pos_embed(posemb, posemb_new, hight, width):
        # Rescale the grid of position embeddings when loading from state_dict. Adapted from
        # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
        ntok_new = posemb_new.shape[1]

        posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1

        gs_old = int(math.sqrt(len(posemb_grid)))
        print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
        posemb = torch.cat([posemb_token, posemb_grid], dim=1)
        return posemb

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
