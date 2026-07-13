from user import User, norm
from config import device, batch_size

class LabelPoison(User):
    def train(self):
        losses = []
        correct_pred = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.dataloader):
            if batch_idx == batch_size:
                break

            images = images.to(device)
            labels = labels.to(device)
            
            poisoned_labels = labels.clone()

            poisoned_labels[:] = (labels + 1) % 10 # Shift every label to the next class



            pred = self.model(norm(images))
            pred_labels = pred.argmax(dim=1)
            loss = self.loss_fn(pred, poisoned_labels)

            correct_pred += (pred_labels == poisoned_labels).sum().item()
            total += labels.size(0)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            losses.append(loss.item())

        p_acc = round((correct_pred/total), 2)
        avg_loss = round(sum(losses)/len(losses), 2)

        return avg_loss, p_acc


class WeightMan(User):
    def get_weight(self):
        weights = {
            k: v.clone()
            for k, v in self.model.state_dict().items()
        }

        for key in weights:
            if weights[key].dtype == torch.float32:
                weights[key] *= 2

        return weights


class SignFlip(User):
    def get_weight(self):
        weights = {
            k: v.clone()
            for k, v in self.model.state_dict().items()
        }
        
        for key in weights:
            if weihgts[key].dtype == torch.float32:
                update = (weights[key] - self.global_weight[key])
                mal_update  = -3 * update
                weights[key] = (self.global_weights[key] + mal_update)

        return weights
                
