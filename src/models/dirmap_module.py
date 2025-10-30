import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
from PIL import Image
from typing import Callable, Tuple, Dict, Any
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.aggregation import MinMetric

# ----------------------------------------------------------------------------
# --- Definições das Novas Classes de Loss ---
# ----------------------------------------------------------------------------

class WeightedOrientationLoss(nn.Module):
    def __init__(self, lambda_pos: float = 1.0, lambda_neg: float = 0.25):
        super().__init__()
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Raw output from the model (B, N, H, W).
                                   DO NOT apply sigmoid beforehand.
            target (torch.Tensor): Ground truth labels (B, 1, H, W).
            mask (torch.Tensor): ROI mask (B, 1, H, W).
        """
        num_classes = logits.shape[1]
        
        # Create one-hot target
        target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # Use log_sigmoid for numerical stability
        # log(p) = log(sigmoid(logits)) = log_sigmoid(logits)
        log_p = F.logsigmoid(logits)
        
        # log(1-p) = log(1 - sigmoid(logits)) = log_sigmoid(-logits)
        log_one_minus_p = F.logsigmoid(-logits)

        # Calculate weighted loss terms
        loss_pos = self.lambda_pos * target_one_hot * log_p
        loss_neg = self.lambda_neg * (1 - target_one_hot) * log_one_minus_p
        
        loss = -(loss_pos + loss_neg)

        # Apply mask and normalize by the number of pixels in the ROI
        mask = mask.float()
        masked_loss = loss * mask
        
        num_roi_pixels = mask.sum()
        if num_roi_pixels > 0:
            total_loss = masked_loss.sum() / num_roi_pixels
        else:
            total_loss = torch.tensor(0.0, device=logits.device)

        return total_loss

# ----------------------------------------------------------------------------

class OrientationCoherenceLoss(nn.Module):
    """
    Implementa a loss de coerência de orientação (L_odpi) do artigo.
    
    L_odpi = |ROI| / (sum_{ROI} Coh) - 1
    
    Onde Coh é o mapa de coerência calculado dos vetores de orientação preditos.
    """
    def __init__(self, N: int = 90, epsilon: float = 1e-8):
        """
        Args:
            N (int): O número de ângulos de orientação discretos.
            epsilon (float): Valor pequeno para estabilidade numérica.
        """
        super().__init__()
        self.N = N
        self.epsilon = epsilon

        # Pré-computa ângulos e o kernel 3x3 (J_3)
        angle_step_deg = 180.0 / N
        angles_deg = torch.arange(N, dtype=torch.float32) * angle_step_deg
        
        # A fórmula usa cos(2 * angulo) e sin(2 * angulo)
        angles_rad_doubled = torch.deg2rad(2 * angles_deg)

        # Registra como 'parameter' (buffers não-treináveis)
        self.cos_terms = nn.Parameter(torch.cos(angles_rad_doubled).view(1, N, 1, 1), requires_grad=False)
        self.sin_terms = nn.Parameter(torch.sin(angles_rad_doubled).view(1, N, 1, 1), requires_grad=False)
        self.j3_kernel = nn.Parameter(torch.ones((1, 1, 3, 3), dtype=torch.float32), requires_grad=False)

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction (torch.Tensor): Mapa de probabilidade do modelo. (B, N, H, W).
            mask (torch.Tensor): Máscara ROI. (B, 1, H, W).
                                 
        Returns:
            torch.Tensor: O valor escalar da loss.
        """

        prediction = torch.sigmoid(logits)
        # --- 1. Computa o vetor de orientação médio d_bar ---
        d_bar_cos = torch.sum(prediction * self.cos_terms, dim=1, keepdim=True) / self.N
        d_bar_sin = torch.sum(prediction * self.sin_terms, dim=1, keepdim=True) / self.N
        
        # --- 2. Calcula Coh = (d_bar * J_3) / (|d_bar| * J_3) ---
        
        # Numerador: Convolui componentes do vetor e calcula magnitude do resultado
        summed_d_cos = F.conv2d(d_bar_cos, self.j3_kernel, padding='same')
        summed_d_sin = F.conv2d(d_bar_sin, self.j3_kernel, padding='same')
        numerator = torch.sqrt(summed_d_cos**2 + summed_d_sin**2 + self.epsilon)
        
        # Denominador: Calcula magnitude dos vetores, depois convolui (média)
        d_bar_mag = torch.sqrt(d_bar_cos**2 + d_bar_sin**2 + self.epsilon)
        denominator = F.conv2d(d_bar_mag, self.j3_kernel, padding='same')
        
        coh = numerator / (denominator + self.epsilon)
        
        # --- 3. Computa a loss final L_odpi ---
        mask = mask.float()
        roi_size = torch.sum(mask)
        
        if roi_size > 0:
            sum_coh_roi = torch.sum(coh * mask)
            loss = roi_size / (sum_coh_roi + self.epsilon) - 1.0
        else:
            loss = torch.tensor(0.0, device=prediction.device, requires_grad=True)
            
        return loss

# ----------------------------------------------------------------------------
# --- Módulo Lightning Atualizado ---
# ----------------------------------------------------------------------------

class EnhancerLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer_dirmap: Callable,
        optimizer_enh: Callable,
        scheduler_dirmap: Callable,
        scheduler_enh: Callable,
        compile: bool,
        output_path: str = None,
        patch_size: Tuple[int, int] = (128, 128),
        use_patches: bool = False,
        stride: int = 8,
        warmup_epochs_dirmap: int = 2,
        # --- NOVOS Hiperparâmetros para a Loss ---
        w_ori: float = 1.0,         # Peso para a loss de orientação ponderada
        w_coh: float = 0.5,         # Peso para a loss de coerência
        lambda_pos: float = 1.0,    # Peso positivo para WeightedOrientationLoss
        lambda_neg: float = 0.25,   # Peso negativo para WeightedOrientationLoss
        N_ori: int = 90             # Número de classes de orientação
    ) -> None:
        super().__init__()
        # Salva todos os HPs, incluindo os novos
        self.save_hyperparameters(logger=False, ignore=["net"]) 
        self.net = net

        self.automatic_optimization = False

        self.mse_criterion = torch.nn.functional.mse_loss
        self.bce_criterion = torch.nn.functional.binary_cross_entropy_with_logits
        
        # --- NOVAS Funções de Loss ---
        self.orientation_loss_fn = WeightedOrientationLoss(
            lambda_pos=self.hparams.lambda_pos, 
            lambda_neg=self.hparams.lambda_neg
        )
        self.coherence_loss_fn = OrientationCoherenceLoss(N=self.hparams.N_ori)
        
        # --- Métricas ---
        self.train_loss = MeanMetric()
        self.train_ori_loss = MeanMetric()
        self.train_enh_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_ori_loss = MeanMetric()
        self.val_enh_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_best = MinMetric()
        
        # --- Outros Atributos ---
        self.patch_size = patch_size
        self.use_patches = use_patches
        self.stride = stride
        self.output_path = output_path


    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)
    
    def on_train_start(self) -> None:
        self.val_loss.reset(); self.val_ori_loss.reset(); self.val_enh_loss.reset(); self.val_loss_best.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, y_dirmap, y_orig, y_bin = batch
        
        # --- Forward ---
        # pred_dirmap são os logits (B, 90, H, W)
        pred_dirmap, pred_enh = self.forward(x)
        
        # --- Preparar Predições ---
        pred_orig, pred_bin = pred_enh[:,0,:,:], pred_enh[:,1,:,:]
        # Converte logits do dirmap para probabilidades (necessário para as novas losses)
        # pred_dirmap_probs = torch.sigmoid(pred_dirmap)

        # --- Preparar Targets ---
        true_orig, true_bin = y_orig[:, 0, :, :], y_bin[:, 0, :, :]
        true_dirmap = F.interpolate(y_dirmap, size=true_bin.shape[1:], mode="bilinear", align_corners=False)
        # Target labels (B, H, W)
        true_dirmap_idx = true_dirmap.argmax(dim=1)

        
        # --- Preparar Inputs para Loss de Orientação ---
        # Labels precisam ter dimensão de canal: (B, H, W) -> (B, 1, H, W)
        true_dirmap_labels = true_dirmap_idx.unsqueeze(1)
        # Usar 'true_dirmap_labels==90' como a máscara ROI: (B, H, W) -> (B, 1, H, W)
        roi_mask = (true_dirmap_labels != 90).long()

        # remove values de máscara
        true_dirmap_labels[true_dirmap_labels == 90] = 0

        # --- Calcular Losses ---
        
        # 1. Loss de Orientação (NOVA LÓGICA)
        loss_ori_weighted = self.orientation_loss_fn(pred_dirmap, true_dirmap_labels, roi_mask)
        loss_ori_coherence = self.coherence_loss_fn(pred_dirmap, roi_mask)
        
        # Soma ponderada dos componentes da loss de orientação
        ori_loss = (self.hparams.w_ori * loss_ori_weighted) + \
                   (self.hparams.w_coh * loss_ori_coherence)
        
        # 3. Loss Total (Inalterada)
        total_loss = ori_loss 
        
        return {"total_loss": ori_loss}
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # return loss or backpropagation will fail
        return loss
    
    def on_train_epoch_start(self):
        """
        Hook executado no início de cada época de treinamento.
        Ideal para congelar/descongelar camadas.
        """
        pass

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        # self.test_acc(preds, targets)
        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def save_orientation_field(self, output_tensor: torch.Tensor, mask:np.ndarray, png_path: str, dir_path: str):
        """
        Save the orientation field from a 90-channel output tensor to a .png image and .dir text file.

        Parameters
        ----------
        output_tensor : torch.Tensor
            Tensor of shape [1, 90, H, W] with probabilities or responses per angle.
        png_path : str
            Path to save the .png image.
        dir_path : str
            Path to save the .dir text file.
        """
        # Ensure tensor is detached and on CPU
        output = output_tensor.cpu().squeeze(0)  # -> [90, H, W]
        assert output.shape[0] == 90, "Output must have 91 channels"

        # Find channel with maximum response per pixel
        max_indices = torch.argmax(output, dim=0).numpy()  # -> [H, W]

        # Convert channel indices to angles (0°, 2°, ..., 178°)
        angles = max_indices * 2  # [H, W] int

        # Map 180° to -1 for background
        background = -np.ones_like(angles)

        if mask != None:
            angles = np.where(mask == 0, background, angles)

        H, W = angles.shape

        # --- Save PNG using PIL ---
        # Scale angles (0..178) → (0..255) for visualization
        img_array = (angles.astype(np.float32) * (255.0 / 178.0)).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")
        img.save(png_path)

        # --- Save .dir file with multiple columns per line ---
        with open(dir_path, "w") as f:
            f.write(f"{W} {H}\n")  # width and height
            for y in range(H):
                row_values = " ".join(str(angles[y, x]) for x in range(W))
                f.write(row_values + "\n")

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single prediction step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        data  = batch[0]
        names = batch[1]
        x, y = batch

        # print(data.shape) # (28,1,128,128)
        # print(y) # tupla com todos os nomes das imagens do batch

        gabor_path = os.path.join(self.output_path, "gabor")
        if not os.path.exists(gabor_path):
            os.makedirs(gabor_path)

        bin_path = os.path.join(self.output_path, "bin")
        if not os.path.exists(bin_path):
            os.makedirs(bin_path)

        enh_path = os.path.join(self.output_path, "enh")
        if not os.path.exists(enh_path):
            os.makedirs(enh_path)

        dirmap_path = os.path.join(self.output_path, "dirmap")
        if not os.path.exists(dirmap_path):
            os.makedirs(dirmap_path)

        dirmap_png_path = os.path.join(self.output_path, "dirmap_png")
        if not os.path.exists(dirmap_png_path):
            os.makedirs(dirmap_png_path)

        seg_path = os.path.join(self.output_path, "mask")
        if not os.path.exists(seg_path):
            os.makedirs(seg_path)

        
        dirmap_pred, latent_enh = self.forward(x)
 

        for i, name in enumerate(names):

            gabor    = latent_enh[i, 0, :, :]

            gabor = gabor.cpu().numpy()

            gabor = (255 * (gabor - np.min(gabor))/(np.max(gabor) - np.min(gabor))).astype('uint8')

            gabor = Image.fromarray(gabor)
            gabor.save(gabor_path + '/' + name + '.png')
         

            dirmap   = dirmap_pred[i, :, :, :]
            dirmap   = torch.nn.functional.sigmoid(dirmap)

            
            self.save_orientation_field(dirmap, None, f"{dirmap_png_path}/{name}.png", f"{dirmap_path}/{name}.dir")


    # <-------------------------------------------------------------------------------------->