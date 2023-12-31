
"""Synchronized Cross lib Batch Normalization functions"""
import sys
sys.path.append('../')
import torch
import torch.cuda.comm as comm
from torch.autograd import Function
from torch.autograd.function import once_differentiable

if torch.cuda.device_count() > 0:
    from syncbn import lib

__all__ = ['moments', 'syncbatchnorm', 'inp_syncbatchnorm','dist_syncbatchnorm']

class moments_(Function):
    @staticmethod
    def forward(ctx, x):
        if x.is_cuda:
            ex, ex2 = lib.expectation_forward(x)
        else:
            raise NotImplemented
        ctx.save_for_backward(x)
        return ex, ex2

    @staticmethod
    def backward(ctx, dex, dex2):
        x, = ctx.saved_tensors
        if dex.is_cuda:
            dx = lib.expectation_backward(x, dex, dex2)
        else:
            raise NotImplemented
        return dx

class syncbatchnorm_(Function):
    @classmethod
    def forward(cls, ctx, x, gamma, beta, running_mean, running_var,
                extra, sync=True, training=True, momentum=0.1, eps=1e-05,
                activation="none", slope=0.01):
        # save context
        cls._parse_extra(ctx, extra)
        ctx.sync = sync
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        assert activation == 'none'

        # continous inputs
        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()

        if ctx.training:
            if x.is_cuda:
                _ex, _exs = lib.expectation_forward(x)
            else:
                raise NotImplemented

            if ctx.sync:
                if ctx.is_master:
                    _ex, _exs = [_ex.unsqueeze(0)], [_exs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _ex_w, _exs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _ex.append(_ex_w.unsqueeze(0))
                        _exs.append(_exs_w.unsqueeze(0))

                    _ex = comm.gather(_ex).mean(0)
                    _exs = comm.gather(_exs).mean(0)

                    tensors = comm.broadcast_coalesced((_ex, _exs), [_ex.get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_ex, _exs))
                    _ex, _exs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()

            # Update running stats
            _var = _exs - _ex ** 2
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * _ex)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * _var)

            # Mark in-place modified tensors
            ctx.mark_dirty(running_mean, running_var)
        else:
            _ex, _var = running_mean.contiguous(), running_var.contiguous()
            _exs = _var + _ex ** 2 

        # BN forward + activation
        if x.is_cuda:
            y = lib.batchnorm_forward(x, _ex, _exs, gamma, beta, ctx.eps)

        # Output
        ctx.save_for_backward(x, _ex, _exs, gamma, beta)

        ctx.mark_non_differentiable(running_mean, running_var)
        return y, running_mean, running_var

    @staticmethod
    @once_differentiable
    def backward(ctx, dz, _drunning_mean, _drunning_var):
        x, _ex, _exs, gamma, beta = ctx.saved_tensors
        dz = dz.contiguous()

        # BN backward
        if dz.is_cuda:
            dx, _dex, _dexs, dgamma, dbeta = \
            lib.batchnorm_backward(dz, x, _ex, _exs, gamma, beta, ctx.eps)

        if ctx.training:
            if ctx.sync:
                if ctx.is_master:
                    _dex, _dexs = [_dex.unsqueeze(0)], [_dexs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _dex_w, _dexs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _dex.append(_dex_w.unsqueeze(0))
                        _dexs.append(_dexs_w.unsqueeze(0))

                    _dex = comm.gather(_dex).mean(0)
                    _dexs = comm.gather(_dexs).mean(0)

                    tensors = comm.broadcast_coalesced((_dex, _dexs), [_dex.get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_dex, _dexs))
                    _dex, _dexs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()

            if x.is_cuda:
                dx_ = lib.expectation_backward(x, _dex, _dexs)
            else:
                raise NotImplemented
            dx = dx + dx_

        return dx, dgamma, dbeta, None, None, None, None, None, None, None, None, None

    @staticmethod
    def _parse_extra(ctx, extra):
        ctx.is_master = extra["is_master"]
        if ctx.is_master:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queues = extra["worker_queues"]
            ctx.worker_ids = extra["worker_ids"]
        else:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queue = extra["worker_queue"]

def _act_forward(ctx, x):
    if ctx.activation.lower() == "leaky_relu":
        if x.is_cuda:
            lib.leaky_relu_forward(x, ctx.slope)
        else:
            raise NotImplemented
    else:
        assert ctx.activation == 'none'

def _act_backward(ctx, x, dx):
    if ctx.activation.lower() == "leaky_relu":
        if x.is_cuda:
            lib.leaky_relu_backward(x, dx, ctx.slope)
        else:
            raise NotImplemented
    else:
        assert ctx.activation == 'none'

class inp_syncbatchnorm_(Function):
    @classmethod
    def forward(cls, ctx, x, gamma, beta, running_mean, running_var,
                extra, sync=True, training=True, momentum=0.1, eps=1e-05,
                activation="none", slope=0.01):
        # save context
        cls._parse_extra(ctx, extra)
        ctx.sync = sync
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope

        # continous inputs
        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()

        if ctx.training:
            if x.is_cuda:
                _ex, _exs = lib.expectation_forward(x)
            else:
                raise NotImplemented

            if ctx.sync:
                if ctx.is_master:
                    _ex, _exs = [_ex.unsqueeze(0)], [_exs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _ex_w, _exs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _ex.append(_ex_w.unsqueeze(0))
                        _exs.append(_exs_w.unsqueeze(0))

                    _ex = comm.gather(_ex).mean(0)
                    _exs = comm.gather(_exs).mean(0)

                    tensors = comm.broadcast_coalesced((_ex, _exs), [_ex.get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_ex, _exs))
                    _ex, _exs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()

            # Update running stats
            _var = _exs - _ex ** 2
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * _ex)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * _var)

            # Mark in-place modified tensors
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            _ex, _var = running_mean.contiguous(), running_var.contiguous()
            _exs = _var + _ex ** 2 
            ctx.mark_dirty(x)

        # BN forward + activation
        if x.is_cuda:
         lib.batchnorm_inp_forward(x, _ex, _exs, gamma, beta, ctx.eps)
        else:
            raise NotImplemented

        _act_forward(ctx, x)

        # Output
        ctx.save_for_backward(x, _ex, _exs, gamma, beta)

        ctx.mark_non_differentiable(running_mean, running_var)
        return x, running_mean, running_var

    @staticmethod
    @once_differentiable
    def backward(ctx, dz, _drunning_mean, _drunning_var):
        z, _ex, _exs, gamma, beta = ctx.saved_tensors
        dz = dz.contiguous()

        # Undo activation
        _act_backward(ctx, z, dz)

        # BN backward
        if dz.is_cuda:
            dx, _dex, _dexs, dgamma, dbeta = \
             lib.batchnorm_inp_backward(dz, z, _ex, _exs, gamma, beta, ctx.eps)
        else:
            raise NotImplemented

        if ctx.training:
            if ctx.sync:
                if ctx.is_master:
                    _dex, _dexs = [_dex.unsqueeze(0)], [_dexs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _dex_w, _dexs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _dex.append(_dex_w.unsqueeze(0))
                        _dexs.append(_dexs_w.unsqueeze(0))

                    _dex = comm.gather(_dex).mean(0)
                    _dexs = comm.gather(_dexs).mean(0)

                    tensors = comm.broadcast_coalesced((_dex, _dexs), [_dex.get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_dex, _dexs))
                    _dex, _dexs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()

            if z.is_cuda:
             lib.expectation_inp_backward(dx, z, _dex, _dexs, _ex, _exs, gamma, beta, ctx.eps)
            else:
                raise NotImplemented

        return dx, dgamma, dbeta, None, None, None, None, None, None, None, None, None

    @staticmethod
    def _parse_extra(ctx, extra):
        ctx.is_master = extra["is_master"]
        if ctx.is_master:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queues = extra["worker_queues"]
            ctx.worker_ids = extra["worker_ids"]
        else:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queue = extra["worker_queue"]

class dist_syncbatchnorm_(Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, running_mean, running_var, eps, momentum, training, process_group):
        x = x.contiguous()
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.process_group = process_group

        if not ctx.training:
            _ex, _var = running_mean.contiguous(), running_var.contiguous()
            _exs = _var + _ex ** 2 
            if x.is_cuda:
                y = lib.batchnorm_forward(x, _ex, _exs, gamma, beta, ctx.eps)
            ctx.save_for_backward(x, _ex, _exs, gamma, beta)
            return y

        size = x.numel() // x.size(1)
        if size == 1:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))

        if x.is_cuda:
            _ex, _exs = lib.expectation_forward(x)
        else:
            raise NotImplemented

        count = torch.Tensor([1]).to(x.device)
        count_all_reduce = torch.distributed.all_reduce(count, group=process_group, async_op=True)
        _ex_all_reduce = torch.distributed.all_reduce(_ex, group=process_group, async_op=True)
        _exs_all_reduce = torch.distributed.all_reduce(_exs, group=process_group, async_op=True)

        count_all_reduce.wait()
        _ex_all_reduce.wait()
        _exs_all_reduce.wait()

        _ex = _ex / count
        _exs = _exs / count

        # Update running stats
        _var = _exs - _ex ** 2
        running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * _ex)
        running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * _var)

        # Mark in-place modified tensors
        ctx.mark_dirty(running_mean, running_var)

        # BN forward + activation
        if x.is_cuda:
            y = lib.batchnorm_forward(x, _ex, _exs, gamma, beta, ctx.eps)

        ctx.save_for_backward(x, _ex, _exs, gamma, beta)
        return y

    @staticmethod
    def backward(ctx, dz):
        x, _ex, _exs, gamma, beta = ctx.saved_tensors
        dz = dz.contiguous()

        # BN backward
        if dz.is_cuda:
            dx, _dex, _dexs, dgamma, dbeta = \
                lib.batchnorm_backward(dz, x, _ex, _exs, gamma, beta, ctx.eps)
        else:
            raise NotImplemented

        if ctx.training:
            process_group = ctx.process_group
            count = torch.Tensor([1]).to(x.device)
            count_all_reduce = torch.distributed.all_reduce(count, group=process_group, async_op=True)
            _dex_all_reduce = torch.distributed.all_reduce(_dex, group=process_group, async_op=True)
            _dexs_all_reduce = torch.distributed.all_reduce(_dexs, group=process_group, async_op=True)

            count_all_reduce.wait()
            _dex_all_reduce.wait()
            _dexs_all_reduce.wait()

            _dex = _dex / count
            _dexs = _dexs / count

            if x.is_cuda:
                dx_ = lib.expectation_backward(x, _dex, _dexs)
            else:
                raise NotImplemented
            dx = dx + dx_

        return dx, dgamma, dbeta, None, None, None, None, None, None
    
moments = moments_.apply
syncbatchnorm = syncbatchnorm_.apply
inp_syncbatchnorm = inp_syncbatchnorm_.apply
dist_syncbatchnorm = dist_syncbatchnorm_.apply