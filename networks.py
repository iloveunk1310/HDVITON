import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.utils.spectral_norm import spectral_norm

class BaseNetwork(nn.Module):
    def __init__(self):
        """
        Lớp nền trừu tượng cho tất cả các mạng trong VITON-HD.
        Cung cấp các tiện ích chung: in thông tin mạng và khởi tạo trọng số.
        """
        super(BaseNetwork, self).__init__()

    def print_network(self):
        """
        In tên lớp và tổng số tham số (triệu) của mạng ra màn hình.

        Input:
            Không có tham số đầu vào (dùng self.parameters() nội bộ).

        """
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print("Network [{}] was created. Total number of parameters: {:.1f} million. "
              "To see the architecture, do print(network).".format(self.__class__.__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        """
        Khởi tạo trọng số cho toàn bộ các lớp Conv, Linear và BatchNorm2d trong mạng
        theo phương pháp được chỉ định.

        Input:
            init_type : Phương pháp khởi tạo trọng số. Các giá trị hợp lệ:
                        'normal'         - phân phối chuẩn N(0, gain)
                        'xavier'         - Xavier normal
                        'xavier_uniform' - Xavier uniform
                        'kaiming'        - Kaiming normal (phù hợp với ReLU)
                        'orthogonal'     - ma trận trực giao
                        'none'           - dùng mặc định của PyTorch
            gain      : Hệ số khuếch đại độ lệch chuẩn dùng trong một số phương pháp khởi tạo.

        Output:
            Không có giá trị trả về. Sửa trọng số của mạng in-place.
        """
        def init_func(m):
            """
            Hàm nội bộ được áp dụng cho từng module con.

            Input:
                m : Module con (Conv2d, Linear, BatchNorm2d, ...) cần khởi tạo trọng số.

            Output:
                Không có giá trị trả về. Sửa m.weight và m.bias in-place.
            """
            classname = m.__class__.__name__
            if 'BatchNorm2d' in classname:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif ('Conv' in classname or 'Linear' in classname) and hasattr(m, 'weight'):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError("initialization method '{}' is not implemented".format(init_type))
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, *inputs):
        """
        Phương thức forward trừu tượng, cần được ghi đè ở lớp con.

        Input:
            *inputs : Các tensor đầu vào tùy ý (tùy lớp con định nghĩa).

        """
        pass


# -------------------------------------------------- SegGenerator-related classes --------------------------------------------
class SegGenerator(BaseNetwork):
    def __init__(self, opt, input_nc, output_nc=13, norm_layer=nn.InstanceNorm2d):
        """
        Xây dựng kiến trúc UNet 5 tầng để dự đoán bản đồ phân đoạn ngữ nghĩa (segmentation map)
        của người sau khi thay quần áo. Encoder giảm dần độ phân giải qua MaxPool,
        Decoder khôi phục qua Upsample kết hợp skip connection từ encoder.

        Input:
            opt        : Đối tượng chứa các tham số cấu hình (init_type, init_variance).
            input_nc   : Số kênh đầu vào (clothing-agnostic representation + cloth image + pose map).
            output_nc  : Số lớp phân đoạn đầu ra (mặc định 13 class ngữ nghĩa).
            norm_layer : Lớp chuẩn hóa sử dụng trong các khối conv (mặc định InstanceNorm2d).

        Output:
            Không có giá trị trả về. Khởi tạo các lớp conv1–conv9, up6–up9,
            MaxPool2d, Dropout và Sigmoid.
        """
        super(SegGenerator, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU(),
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU(),
                                   nn.Conv2d(128, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU(),
                                   nn.Conv2d(256, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU())

        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1), norm_layer(1024), nn.ReLU(),
                                   nn.Conv2d(1024, 1024, kernel_size=3, padding=1), norm_layer(1024), nn.ReLU())

        self.up6 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv2d(1024, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU())

        self.up7 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv2d(512, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU(),
                                   nn.Conv2d(256, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU())

        self.up8 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv2d(256, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU(),
                                   nn.Conv2d(128, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU())

        self.up9 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv2d(128, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU())
        self.conv9 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU(),
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU(),
                                   nn.Conv2d(64, output_nc, kernel_size=3, padding=1))

        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

        self.print_network()
        self.init_weights(opt.init_type, opt.init_variance)

    def forward(self, x):
        """
        Thực hiện forward pass qua UNet: encoder (conv1→conv5 với pooling và dropout)
        rồi decoder (up6→conv9 với skip connection từ encoder tương ứng).

        Input:
            x : Tensor đầu vào, shape (B, input_nc, H, W).
                Gồm clothing-agnostic person representation, ảnh quần áo và pose map
                được ghép kênh lại.

        Output:
            Tensor shape (B, output_nc, H, W), giá trị trong [0, 1] sau Sigmoid.
            Mỗi kênh tương ứng xác suất thuộc một class ngữ nghĩa (13 class).
        """
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.drop(self.conv4(self.pool(conv3)))
        conv5 = self.drop(self.conv5(self.pool(conv4)))

        conv6 = self.conv6(torch.cat((conv4, self.up6(conv5)), 1))
        conv7 = self.conv7(torch.cat((conv3, self.up7(conv6)), 1))
        conv8 = self.conv8(torch.cat((conv2, self.up8(conv7)), 1))
        conv9 = self.conv9(torch.cat((conv1, self.up9(conv8)), 1))
        return self.sigmoid(conv9)


# ----------------------------------------------- GMM-related classes ------------------------------------------------

class FeatureExtraction(BaseNetwork):
    def __init__(self, input_nc, ngf=64, num_layers=4, norm_layer=nn.BatchNorm2d):
        """
        Xây dựng mạng CNN trích xuất đặc trưng hình ảnh phục vụ cho module GMM.
        Gồm num_layers lớp Conv stride-2 để giảm dần không gian, tiếp theo là
        2 lớp Conv stride-1 để tinh chỉnh đặc trưng về 512 kênh.

        Input:
            input_nc   : Số kênh ảnh đầu vào.
            ngf        : Số filter ban đầu (tăng gấp đôi qua mỗi tầng, tối đa 512).
            num_layers : Số lớp downsampling stride-2.
            norm_layer : Lớp chuẩn hóa (mặc định BatchNorm2d).

        Output:
            Không có giá trị trả về. Khởi tạo self.model là Sequential chứa
            toàn bộ chuỗi Conv–ReLU–Norm.
        """
        super(FeatureExtraction, self).__init__()

        nf = ngf
        layers = [nn.Conv2d(input_nc, nf, kernel_size=4, stride=2, padding=1), nn.ReLU(), norm_layer(nf)]

        for i in range(1, num_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1), nn.ReLU(), norm_layer(nf)]

        layers += [nn.Conv2d(nf, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(), norm_layer(512)]
        layers += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU()]

        self.model = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x):
        """
        Trích xuất feature map từ ảnh đầu vào qua chuỗi Conv downsampling.

        Input:
            x : Tensor ảnh đầu vào, shape (B, input_nc, H, W).

        Output:
            Tensor feature map, shape (B, 512, H/2^num_layers, W/2^num_layers).
            Mang thông tin ngữ nghĩa dùng để tính tương quan giữa 2 ảnh trong GMM.
        """
        return self.model(x)


class FeatureCorrelation(nn.Module):
    def __init__(self):
        """
        Khởi tạo module tính tương quan chéo (cross-correlation) giữa 2 feature map.
        Không có tham số học được; chỉ thực hiện phép nhân ma trận.
        """
        super(FeatureCorrelation, self).__init__()

    def forward(self, featureA, featureB):
        """
        Tính tương quan chéo giữa feature map của ảnh người (A) và ảnh quần áo (B)
        bằng phép nhân ma trận batch (bmm). Kết quả mã hóa mức độ tương đồng
        giữa từng vị trí không gian của A với toàn bộ vị trí của B.

        Input:
            featureA : Tensor feature của ảnh người (clothing-agnostic), shape (B, C, H, W).
            featureB : Tensor feature của ảnh quần áo, shape (B, C, H, W).

        Output:
            Tensor correlation map, shape (B, H*W, H, W).
            Mỗi vị trí (h, w) trong output chứa vector tương quan độ dài H*W
            biểu diễn sự tương đồng với tất cả vị trí của featureB.
        """
        # Reshape features for matrix multiplication.
        b, c, h, w = featureA.size()
        featureA = featureA.permute(0, 3, 2, 1).reshape(b, w * h, c)
        featureB = featureB.reshape(b, c, h * w)

        # Perform matrix multiplication.
        corr = torch.bmm(featureA, featureB).reshape(b, w * h, h, w)
        return corr


class FeatureRegression(nn.Module):
    def __init__(self, input_nc=512, output_size=6, norm_layer=nn.BatchNorm2d):
        """
        Xây dựng mạng hồi quy dự đoán tham số biến đổi TPS (theta) từ correlation map.
        Gồm chuỗi Conv thu nhỏ không gian tiếp theo là lớp Linear để ra vector tham số.

        Input:
            input_nc    : Số kênh đầu vào (= H*W của feature map dùng làm correlation).
            output_size : Số phần tử của vector theta đầu ra
                          (= 2 * grid_size^2, tương ứng tọa độ x và y của các control point).
            norm_layer  : Lớp chuẩn hóa (mặc định BatchNorm2d).

        Output:
            Không có giá trị trả về. Khởi tạo self.conv (chuỗi Conv) và self.linear (FC layer).
        """
        super(FeatureRegression, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, kernel_size=4, stride=2, padding=1), norm_layer(512), nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1), norm_layer(256), nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU()
        )
        self.linear = nn.Linear(64 * (input_nc // 16), output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Dự đoán vector tham số TPS (theta) từ correlation map đầu vào.

        Input:
            x : Tensor correlation map, shape (B, input_nc, H, W).
                Đầu ra từ FeatureCorrelation, mã hóa sự tương đồng vị trí giữa 2 ảnh.

        Output:
            Tensor theta, shape (B, output_size), giá trị trong [-1, 1] sau Tanh.
            Biểu diễn độ lệch (offset) của các control point TPS so với vị trí lưới đều.
        """
        x = self.conv(x)
        x = self.linear(x.reshape(x.size(0), -1))
        return self.tanh(x)


class TpsGridGen(nn.Module):
    def __init__(self, opt, dtype=torch.float):
        """
        Khởi tạo bộ sinh lưới biến đổi Thin Plate Spline (TPS).
        Tạo lưới tọa độ pixel chuẩn hóa và lưới control point đều,
        tính sẵn ma trận nghịch đảo L^{-1} để dùng khi apply_transformation.

        Input:
            - opt   : Đối tượng cấu hình chứa:
                    opt.load_height - chiều cao ảnh đích (số pixel).
                    opt.load_width  - chiều rộng ảnh đích (số pixel).
                    opt.grid_size   - số control point trên mỗi chiều (lưới đều grid_size × grid_size).
            - dtype : Kiểu dữ liệu tensor (mặc định torch.float).

        Output:
            Đăng ký các buffer:
                grid_X, grid_Y  - lưới tọa độ pixel chuẩn hóa [-0.9, 0.9].
                P_X_base, P_Y_base - tọa độ control point gốc (trước khi dịch chuyển).
                Li              - ma trận nghịch đảo L^{-1} của hệ TPS, shape (1, N+3, N+3).
                P_X, P_Y        - tọa độ control point dạng tensor 5D để broadcast.
        """
        super(TpsGridGen, self).__init__()

        # Create a grid in numpy.
        # TODO: set an appropriate interval ([-1, 1] in CP-VTON, [-0.9, 0.9] in the current version of VITON-HD)
        grid_X, grid_Y = np.meshgrid(np.linspace(-0.9, 0.9, opt.load_width), np.linspace(-0.9, 0.9, opt.load_height))
        grid_X = torch.tensor(grid_X, dtype=dtype).unsqueeze(0).unsqueeze(3)  # size: (1, h, w, 1)
        grid_Y = torch.tensor(grid_Y, dtype=dtype).unsqueeze(0).unsqueeze(3)  # size: (1, h, w, 1)

        # Initialize the regular grid for control points P.
        self.N = opt.grid_size * opt.grid_size
        coords = np.linspace(-0.9, 0.9, opt.grid_size)
        # FIXME: why P_Y and P_X are swapped?
        P_Y, P_X = np.meshgrid(coords, coords)
        P_X = torch.tensor(P_X, dtype=dtype).reshape(self.N, 1)
        P_Y = torch.tensor(P_Y, dtype=dtype).reshape(self.N, 1)
        P_X_base = P_X.clone()
        P_Y_base = P_Y.clone()

        Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
        P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)  # size: (1, 1, 1, 1, self.N)
        P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)  # size: (1, 1, 1, 1, self.N)

        self.register_buffer('grid_X', grid_X, False)
        self.register_buffer('grid_Y', grid_Y, False)
        self.register_buffer('P_X_base', P_X_base, False)
        self.register_buffer('P_Y_base', P_Y_base, False)
        self.register_buffer('Li', Li, False)
        self.register_buffer('P_X', P_X, False)
        self.register_buffer('P_Y', P_Y, False)

    # TODO: refactor
    def compute_L_inverse(self, X, Y):
        """
        Tính nghịch đảo của ma trận hệ phương trình TPS (ma trận L)
        từ tọa độ các control point. L^{-1} được dùng để giải hệ và tìm
        trọng số biến đổi W (non-linear) và A (affine).

        Input:
            X : Tensor tọa độ x của N control point, shape (N, 1).
            Y : Tensor tọa độ y của N control point, shape (N, 1).

        Output:
            Tensor L^{-1}, shape (N+3, N+3).
            Ma trận nghịch đảo dùng để tính trọng số TPS trong apply_transformation.
        """
        N = X.size()[0] # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N,N)
        Ymat = Y.expand(N,N)
        P_dist_squared = torch.pow(Xmat-Xmat.transpose(0,1),2)+torch.pow(Ymat-Ymat.transpose(0,1),2)
        P_dist_squared[P_dist_squared==0]=1 # make diagonal 1 to avoid NaN in log computation
        K = torch.mul(P_dist_squared,torch.log(P_dist_squared))
        # construct matrix L
        O = torch.FloatTensor(N,1).fill_(1)
        Z = torch.FloatTensor(3,3).fill_(0)
        P = torch.cat((O,X,Y),1)
        L = torch.cat((torch.cat((K,P),1),torch.cat((P.transpose(0,1),Z),1)),0)
        Li = torch.inverse(L)
        return Li

    # TODO: refactor
    def apply_transformation(self, theta, points):
        """
        Áp dụng biến đổi TPS lên lưới tọa độ điểm đích dựa trên tham số theta
        (vị trí control point dịch chuyển). Tính tọa độ mới của mỗi pixel
        theo công thức TPS: tổng phần affine và phần non-linear (radial basis).

        Input:
            theta  : Tensor tham số biến đổi TPS, shape (B, 2*N) hoặc (B, 2*N, 1, 1).
                     Nửa đầu là offset tọa độ x, nửa sau là offset tọa độ y của N control point.
            points : Tensor lưới tọa độ nguồn cần biến đổi, shape (B_or_1, H, W, 2).
                     points[:,:,:,0] là tọa độ X, points[:,:,:,1] là tọa độ Y.

        Output:
            Tensor lưới tọa độ đã biến đổi, shape (B, H, W, 2).
            Dùng trực tiếp với F.grid_sample để warp ảnh quần áo.
        """
        if theta.dim()==2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords
        # and points[:,:,:,1] are the Y coords

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X=theta[:,:self.N,:,:].squeeze(3)
        Q_Y=theta[:,self.N:,:,:].squeeze(3)
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1,points_h,points_w,1,self.N))
        P_Y = self.P_Y.expand((1,points_h,points_w,1,self.N))

        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_X)
        W_Y = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_X)
        A_Y = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:,:,:,0].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,0].size()+(1,self.N))
        points_Y_for_summation = points[:,:,:,1].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,1].size()+(1,self.N))

        if points_b==1:
            delta_X = points_X_for_summation-P_X
            delta_Y = points_Y_for_summation-P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation-P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation-P_Y.expand_as(points_Y_for_summation)

        dist_squared = torch.pow(delta_X,2)+torch.pow(delta_Y,2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared==0]=1 # avoid NaN in log computation
        U = torch.mul(dist_squared,torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:,:,:,0].unsqueeze(3)
        points_Y_batch = points[:,:,:,1].unsqueeze(3)
        if points_b==1:
            points_X_batch = points_X_batch.expand((batch_size,)+points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,)+points_Y_batch.size()[1:])

        points_X_prime = A_X[:,:,:,:,0]+ \
                       torch.mul(A_X[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_X[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_X,U.expand_as(W_X)),4)

        points_Y_prime = A_Y[:,:,:,:,0]+ \
                       torch.mul(A_Y[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_Y[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_Y,U.expand_as(W_Y)),4)

        return torch.cat((points_X_prime,points_Y_prime),3)

    def forward(self, theta):
        """
        Sinh lưới tọa độ TPS đã biến đổi từ tham số theta,
        sử dụng lưới pixel chuẩn hóa (grid_X, grid_Y) làm điểm nguồn.

        Input:
            theta : Tensor tham số TPS, shape (B, 2*N).
                    Đầu ra từ FeatureRegression, biểu diễn offset của control point.

        Output:
            Tensor warped_grid, shape (B, H, W, 2).
            Lưới tọa độ nguồn để dùng với F.grid_sample nhằm warp ảnh quần áo.
        """
        warped_grid = self.apply_transformation(theta, torch.cat((self.grid_X, self.grid_Y), 3))
        return warped_grid


class GMM(nn.Module):
    def __init__(self, opt, inputA_nc, inputB_nc):
        """
        Xây dựng module Geometric Matching Module (GMM) - pipeline đầy đủ để
        tính lưới biến đổi TPS từ cặp ảnh (người, quần áo).
        Gồm: FeatureExtraction × 2 → FeatureCorrelation → FeatureRegression → TpsGridGen.

        Input:
            opt       : Đối tượng cấu hình chứa load_height, load_width, grid_size.
            inputA_nc : Số kênh ảnh A (clothing-agnostic person representation).
            inputB_nc : Số kênh ảnh B (cloth image).

        Output:
            Không có giá trị trả về. Khởi tạo extractionA, extractionB,
            correlation, regression và gridGen.
        """
        super(GMM, self).__init__()

        self.extractionA = FeatureExtraction(inputA_nc, ngf=64, num_layers=4)
        self.extractionB = FeatureExtraction(inputB_nc, ngf=64, num_layers=4)
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(input_nc=(opt.load_width // 64) * (opt.load_height // 64),
                                            output_size=2 * opt.grid_size**2)
        self.gridGen = TpsGridGen(opt)

    def forward(self, inputA, inputB):
        """
        Tính tham số TPS và lưới biến đổi từ cặp ảnh (người, quần áo).
        Pipeline: trích xuất feature → chuẩn hóa L2 → tính tương quan →
        hồi quy theta → sinh warped grid.

        Input:
            inputA : Tensor ảnh người (clothing-agnostic), shape (B, inputA_nc, H, W).
            inputB : Tensor ảnh quần áo, shape (B, inputB_nc, H, W).

        Output:
            theta       : Tensor tham số biến đổi TPS, shape (B, 2*grid_size^2).
                          Dùng để phân tích mức độ biến dạng.
            warped_grid : Tensor lưới tọa độ, shape (B, H, W, 2).
                          Dùng với F.grid_sample để warp ảnh quần áo về đúng hình dáng cơ thể.
        """
        featureA = F.normalize(self.extractionA(inputA), dim=1)
        featureB = F.normalize(self.extractionB(inputB), dim=1)
        corr = self.correlation(featureA, featureB)
        theta = self.regression(corr)

        warped_grid = self.gridGen(theta)
        return theta, warped_grid


#     ------------ ALIASGenerator-related classes --------------------------------------------
class MaskNorm(nn.Module):
    def __init__(self, norm_nc):
        """
        Xây dựng lớp chuẩn hóa theo mask: chuẩn hóa vùng foreground (quần áo)
        và background riêng biệt trước khi tổng hợp lại.
        Dùng InstanceNorm2d không có affine làm chuẩn hóa cơ sở.

        Input:
            norm_nc : Số kênh feature cần chuẩn hóa.

        Output:
            Không có giá trị trả về. Khởi tạo self.norm_layer (InstanceNorm2d).
        """
        super(MaskNorm, self).__init__()

        self.norm_layer = nn.InstanceNorm2d(norm_nc, affine=False)

    def normalize_region(self, region, mask):
        """
        Chuẩn hóa một vùng ảnh (foreground hoặc background) dựa trên số pixel hợp lệ
        trong mask, sau đó scale kết quả theo tỉ lệ diện tích vùng đó.

        Input:
            region : Tensor feature của vùng cần chuẩn hóa (đã nhân với mask),
                     shape (B, C, H, W).
            mask   : Tensor mask nhị phân chỉ định pixel thuộc vùng, shape (B, 1, H, W).
                     Giá trị 1 = pixel thuộc vùng, 0 = không thuộc.

        Output:
            Tensor feature đã chuẩn hóa theo vùng, shape (B, C, H, W).
            Scale theo sqrt(num_pixels / (H*W)) để giữ cân bằng với InstanceNorm toàn cục.
        """
        b, c, h, w = region.size()

        num_pixels = mask.sum((2, 3), keepdim=True)  # size: (b, 1, 1, 1)
        num_pixels[num_pixels == 0] = 1
        mu = region.sum((2, 3), keepdim=True) / num_pixels  # size: (b, c, 1, 1)

        normalized_region = self.norm_layer(region + (1 - mask) * mu)
        return normalized_region * torch.sqrt(num_pixels / (h * w))

    def forward(self, x, mask):
        """
        Chuẩn hóa feature map theo mask bằng cách xử lý foreground và background
        độc lập rồi cộng lại, tránh pha trộn thống kê giữa 2 vùng.

        Input:
            x    : Tensor feature map đầu vào, shape (B, C, H, W).
            mask : Tensor mask nhị phân (foreground = 1), shape (B, 1, H, W).
                   Thường là vùng quần áo warped.

        Output:
            Tensor feature đã chuẩn hóa theo mask, shape (B, C, H, W).
            = normalized_foreground + normalized_background.
        """
        mask = mask.detach()
        normalized_foreground = self.normalize_region(x * mask, mask)
        normalized_background = self.normalize_region(x * (1 - mask), 1 - mask)
        return normalized_foreground + normalized_background


class ALIASNorm(nn.Module):
    def __init__(self, norm_type, norm_nc, label_nc):
        """
        Xây dựng lớp chuẩn hóa ALIAS (Adaptive Local Instance-Aware Semantic Normalization).
        Tương tự SPADE nhưng sử dụng thêm misalign_mask để phân biệt
        vùng quần áo aligned và misaligned, áp dụng chuẩn hóa khác nhau cho từng vùng.
        Thêm noise học được để tăng tính đa dạng cho ảnh sinh ra.

        Input:
            norm_type : Chuỗi chỉ định kiểu chuẩn hóa, dạng 'alias' + loại:
                        'aliasbatch'    - dùng BatchNorm2d cho phần param-free.
                        'aliasinstance' - dùng InstanceNorm2d.
                        'aliasmask'     - dùng MaskNorm (chuẩn hóa theo mask).
            norm_nc   : Số kênh feature cần chuẩn hóa.
            label_nc  : Số kênh của segmentation map đầu vào để sinh gamma/beta.

        Output:
            Không có giá trị trả về. Khởi tạo:
                noise_scale     - vector tham số học scale noise, shape (norm_nc,).
                param_free_norm - lớp chuẩn hóa không tham số (Batch/Instance/MaskNorm).
                conv_shared     - Conv sinh đặc trưng trung gian từ seg map.
                conv_gamma      - Conv sinh hệ số nhân gamma từ seg map.
                conv_beta       - Conv sinh hệ số cộng beta từ seg map.
        """
        super(ALIASNorm, self).__init__()

        self.noise_scale = nn.Parameter(torch.zeros(norm_nc))

        assert norm_type.startswith('alias')
        param_free_norm_type = norm_type[len('alias'):]
        if param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'mask':
            self.param_free_norm = MaskNorm(norm_nc)
        else:
            raise ValueError(
                "'{}' is not a recognized parameter-free normalization type in ALIASNorm".format(param_free_norm_type)
            )

        nhidden = 128
        ks = 3
        pw = ks // 2
        self.conv_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.conv_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.conv_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, seg, misalign_mask=None):
        """
        Chuẩn hóa feature map x theo thông tin ngữ nghĩa từ segmentation map,
        có xét đến vùng misaligned nếu được cung cấp.
        Thêm noise ngẫu nhiên scale học được trước khi chuẩn hóa.

        Input:
            x             : Tensor feature map cần chuẩn hóa, shape (B, norm_nc, H, W).
            seg           : Tensor segmentation map, shape (B, label_nc, H, W).
                            Cung cấp thông tin ngữ nghĩa để học gamma và beta.
            misalign_mask : Tensor mask vùng misaligned (quần áo warped chưa khớp hoàn toàn),
                            shape (B, 1, H, W) hoặc None.
                            Nếu có → dùng MaskNorm; nếu None → dùng chuẩn hóa thông thường.

        Output:
            Tensor feature đã chuẩn hóa và modulate bởi gamma/beta, shape (B, norm_nc, H, W).
            = normalized * (1 + gamma) + beta, trong đó gamma và beta được học từ seg.
        """
        # Part 1. Generate parameter-free normalized activations.
        b, c, h, w = x.size()
        noise = (torch.randn(b, w, h, 1, device=x.device, dtype=x.dtype) * self.noise_scale).transpose(1, 3)

        if misalign_mask is None:
            normalized = self.param_free_norm(x + noise)
        else:
            normalized = self.param_free_norm(x + noise, misalign_mask)

        # Part 2. Produce affine parameters conditioned on the segmentation map.
        actv = self.conv_shared(seg)
        gamma = self.conv_gamma(actv)
        beta = self.conv_beta(actv)

        # Apply the affine parameters.
        output = normalized * (1 + gamma) + beta
        return output


class ALIASResBlock(nn.Module):
    def __init__(self, opt, input_nc, output_nc, use_mask_norm=True):
        """
        Xây dựng Residual Block sử dụng ALIASNorm thay cho BatchNorm/InstanceNorm thông thường.
        Hỗ trợ spectral normalization trên các lớp Conv và learned shortcut
        khi số kênh input/output khác nhau.

        Input:
            opt          : Đối tượng cấu hình chứa:
                           opt.norm_G      - kiểu chuẩn hóa (vd: 'spectralaliasinstance').
                           opt.semantic_nc - số kênh segmentation map.
            input_nc     : Số kênh đầu vào của residual block.
            output_nc    : Số kênh đầu ra của residual block.
            use_mask_norm: Nếu True → dùng 'aliasmask' (MaskNorm) cho các tầng đầu (có misalign).
                           Nếu False → dùng kiểu norm thông thường (các tầng cuối decoder).

        Output:
            Không có giá trị trả về. Khởi tạo conv_0, conv_1 (và conv_s nếu cần),
            norm_0, norm_1 (và norm_s nếu cần), relu.
        """
        super(ALIASResBlock, self).__init__()

        self.learned_shortcut = (input_nc != output_nc)
        middle_nc = min(input_nc, output_nc)

        self.conv_0 = nn.Conv2d(input_nc, middle_nc, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(middle_nc, output_nc, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(input_nc, output_nc, kernel_size=1, bias=False)

        subnorm_type = opt.norm_G
        if subnorm_type.startswith('spectral'):
            subnorm_type = subnorm_type[len('spectral'):]
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        semantic_nc = opt.semantic_nc
        if use_mask_norm:
            subnorm_type = 'aliasmask'
            semantic_nc = semantic_nc + 1

        self.norm_0 = ALIASNorm(subnorm_type, input_nc, semantic_nc)
        self.norm_1 = ALIASNorm(subnorm_type, middle_nc, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = ALIASNorm(subnorm_type, input_nc, semantic_nc)

        self.relu = nn.LeakyReLU(0.2)

    def shortcut(self, x, seg, misalign_mask):
        """
        Tính nhánh shortcut (residual connection) của block.
        Nếu input_nc == output_nc: trả về x nguyên vẹn (identity).
        Nếu khác nhau: chuẩn hóa x qua norm_s rồi chiếu qua conv_s (1×1 Conv).

        Input:
            x            : Tensor feature map đầu vào, shape (B, input_nc, H, W).
            seg          : Tensor segmentation map, shape (B, semantic_nc, H, W).
            misalign_mask: Tensor mask vùng misaligned, shape (B, 1, H, W) hoặc None.

        Output:
            Tensor shortcut, shape (B, output_nc, H, W).
            Được cộng với nhánh residual trong forward để tạo skip connection.
        """
        if self.learned_shortcut:
            return self.conv_s(self.norm_s(x, seg, misalign_mask))
        else:
            return x

    def forward(self, x, seg, misalign_mask=None):
        """
        Thực hiện forward pass của ALIAS Residual Block:
        interpolate seg/mask về cùng kích thước x, tính shortcut và nhánh residual,
        cộng lại để ra output.

        Input:
            x            : Tensor feature map đầu vào, shape (B, input_nc, H, W).
            seg          : Tensor segmentation map ở độ phân giải bất kỳ,
                           sẽ được resize về (H, W) bên trong.
            misalign_mask: Tensor mask vùng quần áo misaligned, shape (B, 1, H', W') hoặc None.
                           Sẽ được resize về (H, W) nếu không None.

        Output:
            Tensor output = shortcut(x) + conv_1(relu(norm_1(conv_0(relu(norm_0(x)))))),
            shape (B, output_nc, H, W).
        """
        seg = F.interpolate(seg, size=x.size()[2:], mode='nearest')
        if misalign_mask is not None:
            misalign_mask = F.interpolate(misalign_mask, size=x.size()[2:], mode='nearest')

        x_s = self.shortcut(x, seg, misalign_mask)

        dx = self.conv_0(self.relu(self.norm_0(x, seg, misalign_mask)))
        dx = self.conv_1(self.relu(self.norm_1(dx, seg, misalign_mask)))
        output = x_s + dx
        return output


class ALIASGenerator(BaseNetwork):
    def __init__(self, opt, input_nc):
        """
        Xây dựng Generator chính của VITON-HD dựa trên kiến trúc SPADE cải tiến với ALIASNorm.
        Nhận ảnh đầu vào ở nhiều độ phân giải (multi-scale), chiếu qua Conv riêng từng scale,
        rồi decode dần lên độ phân giải đích qua các ALIASResBlock và Upsample.

        Input:
            opt      : Đối tượng cấu hình chứa:
                       opt.num_upsampling_layers - 'normal'(5), 'more'(6) hoặc 'most'(7 lần upsample).
                       opt.ngf                  - số filter cơ sở.
                       opt.norm_G               - kiểu chuẩn hóa cho ALIASResBlock.
                       opt.semantic_nc          - số kênh segmentation map.
                       opt.load_height/width    - kích thước ảnh đầu ra.
                       opt.init_type/variance   - tham số khởi tạo trọng số.
            input_nc : Số kênh ảnh đầu vào (warped cloth + agnostic repr + seg map...).

        Output:
            Không có giá trị trả về. Khởi tạo:
                conv_0..conv_7   - Conv chiếu từng scale về 16 kênh (trừ conv_0 → nf*16).
                head_0           - ALIASResBlock ở độ phân giải thấp nhất.
                G_middle_0/1     - ALIASResBlock ở tầng giữa.
                up_0..up_3/4     - ALIASResBlock decoder với upsample.
                conv_img         - Conv cuối sinh ảnh RGB 3 kênh.
        """
        super(ALIASGenerator, self).__init__()
        self.num_upsampling_layers = opt.num_upsampling_layers

        self.sh, self.sw = self.compute_latent_vector_size(opt)

        nf = opt.ngf
        self.conv_0 = nn.Conv2d(input_nc, nf * 16, kernel_size=3, padding=1)
        for i in range(1, 8):
            self.add_module('conv_{}'.format(i), nn.Conv2d(input_nc, 16, kernel_size=3, padding=1))

        self.head_0 = ALIASResBlock(opt, nf * 16, nf * 16)

        self.G_middle_0 = ALIASResBlock(opt, nf * 16 + 16, nf * 16)
        self.G_middle_1 = ALIASResBlock(opt, nf * 16 + 16, nf * 16)

        self.up_0 = ALIASResBlock(opt, nf * 16 + 16, nf * 8)
        self.up_1 = ALIASResBlock(opt, nf * 8 + 16, nf * 4)
        self.up_2 = ALIASResBlock(opt, nf * 4 + 16, nf * 2, use_mask_norm=False)
        self.up_3 = ALIASResBlock(opt, nf * 2 + 16, nf * 1, use_mask_norm=False)
        if self.num_upsampling_layers == 'most':
            self.up_4 = ALIASResBlock(opt, nf * 1 + 16, nf // 2, use_mask_norm=False)
            nf = nf // 2

        self.conv_img = nn.Conv2d(nf, 3, kernel_size=3, padding=1)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

        self.print_network()
        self.init_weights(opt.init_type, opt.init_variance)

    def compute_latent_vector_size(self, opt):
        """
        Tính kích thước không gian của latent vector ở tầng thấp nhất của generator
        dựa trên số lần upsample và kích thước ảnh đích.

        Input:
            opt : Đối tượng cấu hình chứa:
                  opt.num_upsampling_layers - số lần upsample ('normal'=5, 'more'=6, 'most'=7).
                  opt.load_height           - chiều cao ảnh đích (pixel).
                  opt.load_width            - chiều rộng ảnh đích (pixel).

        Output:
            sh : Chiều cao latent vector = load_height // 2^num_up_layers.
            sw : Chiều rộng latent vector = load_width // 2^num_up_layers.
        """
        if self.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif self.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif self.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError("opt.num_upsampling_layers '{}' is not recognized".format(self.num_upsampling_layers))

        sh = opt.load_height // 2**num_up_layers
        sw = opt.load_width // 2**num_up_layers
        return sh, sw

    def forward(self, x, seg, seg_div, misalign_mask):
        """
        Sinh ảnh try-on từ ảnh đầu vào x qua kiến trúc multi-scale decoder với ALIASNorm.
        Ảnh x được resize về 8 độ phân giải khác nhau, mỗi scale được chiếu qua conv tương ứng
        rồi nối (concat) vào feature map tại tầng decoder phù hợp.

        Input:
            x            : Tensor ảnh đầu vào ghép kênh (warped cloth + clothing-agnostic person),
                           shape (B, input_nc, H, W).
            seg          : Tensor segmentation map đầy đủ, shape (B, semantic_nc, H, W).
                           Dùng cho các tầng decoder cuối (không có misalign).
            seg_div      : Tensor segmentation map ghép với misalign_mask,
                           shape (B, semantic_nc+1, H, W).
                           Dùng cho các tầng decoder đầu (có misalign).
            misalign_mask: Tensor mask vùng quần áo warped chưa khớp cơ thể,
                           shape (B, 1, H, W). Dùng trong MaskNorm của ALIASResBlock đầu.

        Output:
            Tensor ảnh try-on sinh ra, shape (B, 3, H, W), giá trị trong [-1, 1] sau Tanh.
            Ảnh RGB biểu diễn người mặc quần áo mới.
        """
        samples = [F.interpolate(x, size=(self.sh * 2**i, self.sw * 2**i), mode='nearest') for i in range(8)]
        features = [self._modules['conv_{}'.format(i)](samples[i]) for i in range(8)]

        x = self.head_0(features[0], seg_div, misalign_mask)

        x = self.up(x)
        x = self.G_middle_0(torch.cat((x, features[1]), 1), seg_div, misalign_mask)
        if self.num_upsampling_layers in ['more', 'most']:
            x = self.up(x)
        x = self.G_middle_1(torch.cat((x, features[2]), 1), seg_div, misalign_mask)

        x = self.up(x)
        x = self.up_0(torch.cat((x, features[3]), 1), seg_div, misalign_mask)
        x = self.up(x)
        x = self.up_1(torch.cat((x, features[4]), 1), seg_div, misalign_mask)
        x = self.up(x)
        x = self.up_2(torch.cat((x, features[5]), 1), seg)
        x = self.up(x)
        x = self.up_3(torch.cat((x, features[6]), 1), seg)
        if self.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(torch.cat((x, features[7]), 1), seg)

        x = self.conv_img(self.relu(x))
        return self.tanh(x)