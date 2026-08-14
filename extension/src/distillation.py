"""
Cross-Modal Distillation — transfer knowledge from RF encoder to acoustic/radar.

The RF encoder (frozen, teacher) produces embeddings that capture drone-ness.
The student encoders (acoustic, radar) learn to produce similar embeddings
without needing RF at inference time.

Loss: L_task + λ * ||z_student - sg(z_teacher)||²

For paired data (TSMS-Drone: radar+RF time-synchronized): direct distillation.
For unpaired data (acoustic): distill using shared drone-type labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    L2 alignment loss between student and teacher embeddings.

    L_distill = λ * ||z_student - stopgrad(z_teacher)||²

    The teacher (RF encoder) is frozen. The student learns to produce
    embeddings that look like RF embeddings, even though it processes
    a different modality.
    """
    def __init__(self, lambda_distill=1.0):
        super().__init__()
        self.lambda_distill = lambda_distill

    def forward(self, z_student, z_teacher, student_labels=None, teacher_labels=None):
        """
        z_student: (B, D) from student encoder (acoustic/radar)
        z_teacher: (B, D) from frozen RF encoder

        If labels are provided, uses label-matched distillation:
        for each student sample, find the teacher sample with the same label
        and align to it. This handles unpaired data.
        """
        if student_labels is not None and teacher_labels is not None:
            # Label-matched distillation (for unpaired data)
            loss = torch.tensor(0.0, device=z_student.device)
            count = 0
            for i in range(len(z_student)):
                label = student_labels[i]
                mask = teacher_labels == label
                if mask.sum() > 0:
                    teacher_match = z_teacher[mask].mean(dim=0)  # centroid of matching teacher
                    loss = loss + F.mse_loss(z_student[i], teacher_match.detach())
                    count += 1
            if count > 0:
                loss = loss / count
        else:
            # Direct distillation (for paired data)
            loss = F.mse_loss(z_student, z_teacher.detach())

        return self.lambda_distill * loss


class CrossModalDistiller:
    """
    Orchestrates cross-modal distillation from RF teacher to student encoders.

    Usage:
        distiller = CrossModalDistiller(rf_encoder, acoustic_encoder, lambda_distill=1.0)
        for batch in paired_loader:
            rf_specs, acoustic_specs, labels = batch
            z_rf = distiller.teacher(rf_specs)
            z_acoustic = distiller.student(acoustic_specs)
            loss = distiller.compute_loss(z_acoustic, z_rf, labels, labels)
    """
    def __init__(self, teacher_encoder, student_encoder, lambda_distill=1.0):
        self.teacher = teacher_encoder
        self.student = student_encoder
        self.distill_loss = DistillationLoss(lambda_distill=lambda_distill)

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def compute_loss(self, z_student, z_teacher, student_labels=None, teacher_labels=None):
        """Returns (total_loss, task_loss, distill_loss)"""
        # Task loss is handled by the student encoder's own compute_loss
        distill = self.distill_loss(z_student, z_teacher, student_labels, teacher_labels)
        return distill
