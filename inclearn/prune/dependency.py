import torch
import torch.nn as nn
import typing
from functools import reduce
from operator import mul
from . import prune
from enum import IntEnum

__all__ = ['PruningPlan', 'Dependency', 'DependencyGraph']

TORCH_CONV = nn.modules.conv._ConvNd
TORCH_BATCHNORM = nn.modules.batchnorm._BatchNorm
TORCH_PRELU = nn.PReLU
TORCH_LINEAR = nn.Linear


class OPTYPE(IntEnum):
    # 枚举类
    CONV = 0
    BN = 1
    LINEAR = 2
    PRELU = 3
    GROUP_CONV = 4

    CONCAT = 5
    SPLIT = 6
    ELEMENTWISE = 7


def _get_module_type(module):
    if isinstance(module, TORCH_CONV):
        if module.groups > 1:
            return OPTYPE.GROUP_CONV
        else:
            return OPTYPE.CONV
    elif isinstance(module, TORCH_BATCHNORM):
        return OPTYPE.BN
    elif isinstance(module, TORCH_PRELU):
        return OPTYPE.PRELU
    elif isinstance(module, TORCH_LINEAR):
        return OPTYPE.LINEAR
    elif isinstance(module, _ConcatOp):
        return OPTYPE.CONCAT
    elif isinstance(module, _SplitOP):
        return OPTYPE.SPLIT
    else:
        return OPTYPE.ELEMENTWISE


def _get_node_out_channel(node):
    if node.type == OPTYPE.CONV or node.type == OPTYPE.GROUP_CONV:
        return node.module.out_channels
    elif node.type == OPTYPE.BN:
        return node.module.num_features
    elif node.type == OPTYPE.LINEAR:
        return node.module.out_features
    elif node.type == OPTYPE.PRELU:
        if node.module.num_parameters == 1:
            return None
        else:
            return node.module.num_parameters
    else:
        return None


def _get_node_in_channel(node):
    if node.type == OPTYPE.CONV or node.type == OPTYPE.GROUP_CONV:
        return node.module.in_channels
    elif node.type == OPTYPE.BN:
        return node.module.num_features
    elif node.type == OPTYPE.LINEAR:
        return node.module.in_features
    elif node.type == OPTYPE.PRELU:
        if node.module.num_parameters == 1:
            return None
        else:
            return node.module.num_parameters
    else:
        return None

# Dummy Pruning fn


def _prune_concat(layer, *args, **kargs):
    return layer, 0


def _prune_split(layer, *args, **kargs):
    return layer, 0


def _prune_elementwise_op(layer, *args, **kargs):
    return layer, 0

# Dummy module


class _ConcatOp(nn.Module):
    def __init__(self):
        super(_ConcatOp, self).__init__()
        self.offsets = None

    def __repr__(self):
        return "_ConcatOp(%s)" % (self.offsets)


class _SplitOP(nn.Module):
    def __init__(self):
        super(_SplitOP, self).__init__()
        self.offsets = None

    def __repr__(self):
        return "_SplitOP(%s)" % (self.offsets)


class _ElementWiseOp(nn.Module):
    def __init__(self):
        super(_ElementWiseOp, self).__init__()

    def __repr__(self):
        return "_ElementWiseOp()"


class _FlattenIndexTransform(object):
    def __init__(self, stride=1, reverse=False):
        self._stride = stride
        self.reverse = reverse

    def __call__(self, idxs):
        new_idxs = []
        if self.reverse == True:
            for i in idxs:
                new_idxs.append(i//self._stride)
                new_idxs = list(set(new_idxs))
        else:
            for i in idxs:
                new_idxs.extend(
                    list(range(i*self._stride, (i+1)*self._stride)))
        return new_idxs


class _ConcatIndexTransform(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs):
        if self.reverse == True:
            new_idxs = [i-self.offset[0]
                        for i in idxs if (i >= self.offset[0] and i < self.offset[1])]
        else:
            new_idxs = [i+self.offset[0] for i in idxs]
        return new_idxs


class _SplitIndexTransform(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs):
        if self.reverse == True:
            new_idxs = [i+self.offset[0] for i in idxs]
        else:
            new_idxs = [i-self.offset[0]
                        for i in idxs if (i >= self.offset[0] and i < self.offset[1])]
        return new_idxs


class Node(object):
    def __init__(self, module, grad_fn, node_name=None):
        self.module = module
        self.grad_fn = grad_fn
        self.inputs = []
        self.outputs = []
        self.dependencies = []
        self._node_name = node_name
        self.type = _get_module_type(module)

    @property
    def node_name(self):
        return "%s (%s)" % (self._node_name, str(self.module)) if self._node_name is not None else str(self.module)

    def add_input(self, node):
        if node not in self.inputs:
            self.inputs.append(node)

    def add_output(self, node):
        if node not in self.outputs:
            self.outputs.append(node)

    def __repr__(self):
        return "<Node: (%s, %s)>" % (self.node_name, self.grad_fn)

    def __str__(self):
        return "<Node: (%s, %s)>" % (self.node_name, self.grad_fn)

    def details(self):
        fmt = "<Node: (%s, %s)>\n" % (self.node_name, self.grad_fn)
        fmt += ' '*4+'IN:\n'
        for in_node in self.inputs:
            fmt += ' '*8+'%s\n' % (in_node)
        fmt += ' '*4+'OUT:\n'
        for out_node in self.outputs:
            fmt += ' '*8+'%s\n' % (out_node)

        fmt += ' '*4+'DEP:\n'
        for dep in self.dependencies:
            fmt += ' '*8+"%s\n" % (dep)
        return fmt


class Dependency(object):
    def __init__(self, trigger, handler, broken_node: Node, index_transform: typing.Callable = None):
        """ Layer dependency in structed neural network pruning. 

        Parameters:
            trigger (Callable or None): a pruning function which will break the dependency 
            handler (Callable): a pruning function to fix the broken dependency
            broken_node (nn.Module): the broken layer
        """
        self.trigger = trigger
        self.handler = handler
        self.broken_node = broken_node
        self.index_transform = index_transform

    def __call__(self, idxs: list, dry_run: bool = False):
        result = self.handler(self.broken_node.module, idxs, dry_run=dry_run)
        return result

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "<DEP: %s => %s on %s>" % ("None" if self.trigger is None else self.trigger.__name__, self.handler.__name__, self.broken_node.node_name)

    def is_triggered_by(self, pruning_fn):
        return pruning_fn == self.trigger

    def __eq__(self, other):
        return ((self.trigger == other.trigger) and
                self.handler == other.handler and
                self.broken_node == other.broken_node)


class PruningPlan(object):
    """ Pruning plan.

    Args:
        dry_run (Callable or None): only return the info about pruning.
        module_to_name (dict): mapping nn.module to a readable name. It will be filled by DependencyGraph.
    """

    def __init__(self):
        self._plans = list()

    def add_plan(self, dep, idxs):
        self._plans.append((dep, idxs))

    @property
    def plan(self):
        return self._plans

    def exec(self, dry_run=False):
        num_pruned = 0
        for dep, idxs in self._plans:
            _, n = dep(idxs, dry_run=dry_run)
            num_pruned += n
        return num_pruned

    def has_dep(self, dep):
        for _dep, _ in self._plans:
            if dep == _dep:
                return True
        return False

    def has_pruning_op(self, dep, idxs):
        for _dep, _idxs in self._plans:
            if _dep.broken_node == dep.broken_node and _dep.handler == dep.handler and _idxs == idxs:
                return True
        return False

    @property
    def is_in_shortcut(self):
        prune_conv_cnt = 0
        for _dep, _idxs in self._plans:
            if _dep.handler.__name__ == 'prune_conv':
                prune_conv_cnt += 1
        if prune_conv_cnt > 1:
            return True
        else:
            return False

    def add_plan_and_merge(self, dep, idxs):
        for i, (_dep, _idxs) in enumerate(self._plans):
            if _dep.broken_node == dep.broken_node and _dep.handler == dep.handler:
                self._plans[i] = (_dep, list(set(_idxs+idxs)))
                return
        self.add_plan(dep, idxs)

    def __str__(self):
        fmt = ""
        fmt += "\n-------------\n"
        totally_pruned = 0
        for dep, idxs in self._plans:
            _, n_pruned = dep(idxs, dry_run=True)
            totally_pruned += n_pruned
            fmt += "[ %s, Index=%s, NumPruned=%d]\n" % (dep, idxs, n_pruned)
        fmt += "%d parameters will be pruned\n" % (totally_pruned)
        fmt += "-------------\n"
        return fmt


class DependencyGraph(object):

    PRUNABLE_MODULES = (nn.modules.conv._ConvNd,
                        nn.modules.batchnorm._BatchNorm, nn.Linear, nn.PReLU)  # 可裁剪的层

    HANDLER = {                         # prune in_channel          # prune out_channel
        OPTYPE.CONV:  (prune.prune_related_conv,   prune.prune_conv),
        OPTYPE.BN:  (prune.prune_batchnorm,      prune.prune_batchnorm),
        OPTYPE.PRELU:  (prune.prune_prelu,          prune.prune_prelu),
        OPTYPE.LINEAR:  (prune.prune_related_linear, prune.prune_linear),
        OPTYPE.GROUP_CONV:  (prune.prune_group_conv,     prune.prune_group_conv),
        OPTYPE.CONCAT:  (_prune_concat,              _prune_concat),
        OPTYPE.SPLIT:  (_prune_split,               _prune_split),
        OPTYPE.ELEMENTWISE:  (_prune_elementwise_op,      _prune_elementwise_op),
    }
    OUTPUT_NODE_RULES = {}
    INPUT_NODE_RULES = {}
    for t1 in HANDLER.keys():
        for t2 in HANDLER.keys():
            # change in_channels of output layer
            OUTPUT_NODE_RULES[(t1, t2)] = (HANDLER[t1][1], HANDLER[t2][0])
            # change out_channels of input layer
            INPUT_NODE_RULES[(t1, t2)] = (HANDLER[t1][0], HANDLER[t2][1])

    def build_dependency(self, model: torch.nn.Module, example_inputs: torch.Tensor, output_transform: callable = None, verbose: bool = True):
        self.verbose = verbose # 显示细节

        self._module_to_name = {module: name for (
            name, module) in model.named_modules()}
        # 获取每层的名称：
        # conv1.weight
        # bn1.weight
        # bn1.bias
        # layer1.0.conv1.weight
        # layer1.0.bn1.weight
        # layer1.0.bn1.bias ...

        # build dependency graph
        self.module_to_node, self.output_grad_fn = self._obtain_forward_graph(
            model, example_inputs, output_transform=output_transform)
        self._build_dependency(self.module_to_node)
        self.update_index()
        return self

    def update_index(self):
        for module, node in self.module_to_node.items():
            if node.type == OPTYPE.LINEAR:
                self._set_fc_index_transform(node)
            if node.type == OPTYPE.CONCAT:
                self._set_concat_index_transform(node)
            if node.type == OPTYPE.SPLIT:
                self._set_split_index_transform(node)

    def get_pruning_plan(self, module, pruning_fn, idxs):

        cur_plan_is_group_conv = False
        if isinstance(module, TORCH_CONV) and module.groups > 1:
            # 只剪枝深度卷积，不剪枝分组卷积
            if module.groups == module.in_channels and module.groups == module.out_channels:
                pruning_fn = prune.prune_group_conv
                cur_plan_is_group_conv = True
            else:
                return None

        self.update_index()
        plan = PruningPlan()
        #  the user pruning operation
        # oot_node = self.module_to_node[module]
        root_node = self.module_to_node.get(module, None)
        if not root_node:
            return None
        # 如果是神经网络的输出层，那么不剪枝
        if root_node.grad_fn in self.output_grad_fn:
            return None

        plan.add_plan(Dependency(pruning_fn, pruning_fn, root_node), idxs)

        visited = set()

        def _fix_denpendency_graph(node, fn, indices):
            visited.add(node)
            for dep in node.dependencies:
                # and dep.broken_node not in visited:
                if dep.is_triggered_by(fn):
                    if dep.index_transform is not None:
                        new_indices = dep.index_transform(indices)
                    else:
                        new_indices = indices

                    if len(new_indices) == 0:
                        continue
                    if dep.broken_node in visited and plan.has_pruning_op(dep, new_indices):
                        continue
                    else:
                        plan.add_plan(dep, new_indices)
                        _fix_denpendency_graph(
                            dep.broken_node, dep.handler, new_indices)

        _fix_denpendency_graph(root_node, pruning_fn, idxs)

        # merge pruning ops
        merged_plan = PruningPlan()
        for dep, idxs in plan.plan:
            merged_plan.add_plan_and_merge(dep, idxs)

        # 如果剪枝计划中有prune_group_conv，但当前节点不是group_conv，则不剪枝，取消计划。
        prune_group_conv_cnt = 0
        for _dep, _idxs in merged_plan._plans:
            if _dep.handler.__name__ == 'prune_group_conv':
                prune_group_conv_cnt += 1
        if prune_group_conv_cnt > 0:
            if not cur_plan_is_group_conv:
                print(4)
                return None

        return merged_plan

    def _build_dependency(self, module_to_node):
        for module, node in module_to_node.items():
            for in_node in node.inputs:
                in_node_rule = self.INPUT_NODE_RULES.get(
                    (node.type, in_node.type), None)
                if in_node_rule is not None:
                    dep = Dependency(
                        trigger=in_node_rule[0], handler=in_node_rule[1], broken_node=in_node)
                    node.dependencies.append(dep)

            for out_node in node.outputs:
                out_node_rule = self.OUTPUT_NODE_RULES.get(
                    (node.type, out_node.type), None)
                if out_node_rule is not None:
                    dep = Dependency(
                        trigger=out_node_rule[0], handler=out_node_rule[1], broken_node=out_node)
                    node.dependencies.append(dep)

    def _obtain_forward_graph(self, model, example_inputs, output_transform):
        # module_to_node = { m: Node( m ) for m in model.modules() if isinstance( m, self.PRUNABLE_MODULES ) }
        model.eval().cpu()
        # Get grad_fn from prunable modules
        grad_fn_to_module = {}
        visited = {}

        def _record_module_grad_fn(module, inputs, outputs):
            # 记录中间层是否有重复使用的
            # 有重复使用的往往和其他层存在依赖关系
            if module not in visited:
                visited[module] = 1
            else:
                visited[module] += 1
            grad_fn_to_module[outputs.grad_fn] = module #

        hooks = [m.register_forward_hook(_record_module_grad_fn) for m in model.modules(
        ) if isinstance(m, self.PRUNABLE_MODULES)] 
        # 获取模型中可裁剪层的输入和输出
        # hook作用：
        #   用来获取某些变量的中间结果的。
        #   Pytorch会自动舍弃图计算的中间结果，所以想要获取这些数值就需要使用hook函数。
        #   hook函数在使用后应及时删除，以避免每次都运行钩子增加运行负载。
        out = model(example_inputs) # 示例输入的输出（包括注册hook的中间层输出）
        for hook in hooks:
            hook.remove() # 删除hook增加运行负载
        reused = [m for (m, count) in visited.items() if count > 1]
        # 创建节点和虚拟模块
        module_to_node = {}
        # 记录神经网络的最后一层，因为这些层不剪枝
        output_grad_fn = []

        def _build_graph(grad_fn, search_final_conv=0):
            # print('grad_fn',grad_fn) grad_fn指向Function对象，用于反向传播的梯度计算之用

            search_final_conv = search_final_conv

            module = grad_fn_to_module.get(grad_fn, None)
            if module is not None and module in module_to_node and module not in reused:
                return module_to_node[module]

            if module is None:
                if not hasattr(grad_fn, 'name'):
                    module = _ElementWiseOp()  # skip customized modules
                    if self.verbose:
                        print(
                            "[Warning] Unrecognized operation: %s. It will be treated as element-wise op" % (str(grad_fn)))
                elif 'catbackward' in grad_fn.name().lower():  # concat op
                    module = _ConcatOp()
                elif 'splitbackward' in grad_fn.name().lower():
                    module = _SplitOP()
                else:
                    module = _ElementWiseOp()   # All other ops are treated as element-wise ops
                grad_fn_to_module[grad_fn] = module  # record grad_fn

            if module not in module_to_node:
                node = Node(module, grad_fn,
                            self._module_to_name.get(module, None))
                module_to_node[module] = node
            else:
                node = module_to_node[module]

            if search_final_conv and grad_fn is not None and hasattr(grad_fn, 'name') and ('MkldnnConvolutionBackward' in grad_fn.name() or 'AddmmBackward' in grad_fn.name()):
                search_final_conv = 0
                output_grad_fn.append(grad_fn)

            if hasattr(grad_fn, 'next_functions'):
                for f in grad_fn.next_functions:
                    # print(f)
                    if f[0] is not None:
                        # skip leaf variables
                        if hasattr(f[0], 'name') and 'accumulategrad' in f[0].name().lower():
                            continue
                        input_node = _build_graph(f[0], search_final_conv)
                        node.add_input(input_node)
                        input_node.add_output(node)
            return node

        if output_transform is not None:
            out = output_transform(out)

        if isinstance(out, (list, tuple)):

            for o in out:
                # print('start1---------------------------------------')
                if isinstance(o, dict):
                    # print('if1---------------------------------------')
                    for key in o:
                        # print('if1---------------------------------------')
                        # print(o[key])
                        # if o[key].grad_fn is not None:
                        if o[key].grad_fn is not None and hasattr(o[key].grad_fn, 'name') and ('MkldnnConvolutionBackward' in o[key].grad_fn.name() or 'AddmmBackward' in o[key].grad_fn.name()):
                            output_grad_fn.append(o[key].grad_fn)
                            _build_graph(o[key].grad_fn, search_final_conv=0)
                        else:
                            _build_graph(o[key].grad_fn, search_final_conv=1)

                elif isinstance(o, (list, tuple)):

                    for new_value in o:
                        # print('if2---------------------------------------')
                        # print(new_value)
                        # if new_value.grad_fn is not None:
                        if new_value.grad_fn is not None and hasattr(new_value.grad_fn, 'name') and ('MkldnnConvolutionBackward' in new_value.grad_fn.name() or 'AddmmBackward' in new_value.grad_fn.name()):
                            output_grad_fn.append(new_value.grad_fn)
                            _build_graph(new_value.grad_fn,
                                         search_final_conv=0)
                        else:
                            _build_graph(new_value.grad_fn,
                                         search_final_conv=1)
                else:
                    # print('if3---------------------------------------')
                    # print(o)
                    # if o.grad_fn is not None:
                    if o.grad_fn is not None and hasattr(o.grad_fn, 'name') and ('MkldnnConvolutionBackward' in o.grad_fn.name() or 'AddmmBackward' in o.grad_fn.name()):
                        output_grad_fn.append(o.grad_fn)
                        _build_graph(o.grad_fn, search_final_conv=0)
                    else:
                        _build_graph(o.grad_fn, search_final_conv=1)

        else:

            if out.grad_fn is not None and hasattr(out.grad_fn, 'name') and ('MkldnnConvolutionBackward' in out.grad_fn.name() or 'AddmmBackward' in out.grad_fn.name()):
                output_grad_fn.append(out.grad_fn)
                _build_graph(out.grad_fn, search_final_conv=0)
            else:
                _build_graph(out.grad_fn, search_final_conv=1)
        return module_to_node, output_grad_fn

    def _set_fc_index_transform(self, fc_node: Node):
        if fc_node.type != OPTYPE.LINEAR:
            return
        visited = set()
        fc_in_features = fc_node.module.in_features
        feature_channels = _get_in_node_out_channels(fc_node.inputs[0])
        stride = fc_in_features // feature_channels
        if stride > 1:
            for in_node in fc_node.inputs:
                for dep in fc_node.dependencies:
                    if dep.broken_node == in_node:
                        dep.index_transform = _FlattenIndexTransform(
                            stride=stride, reverse=True)

                for dep in in_node.dependencies:
                    if dep.broken_node == fc_node:
                        dep.index_transform = _FlattenIndexTransform(
                            stride=stride, reverse=False)

    def _set_concat_index_transform(self, cat_node: Node):
        if cat_node.type != OPTYPE.CONCAT:
            return

        chs = []
        for n in cat_node.inputs:
            chs.append(_get_in_node_out_channels(n))

        offsets = [0]
        for ch in chs:
            offsets.append(offsets[-1]+ch)
        cat_node.module.offsets = offsets

        for i, in_node in enumerate(cat_node.inputs):
            for dep in cat_node.dependencies:
                if dep.broken_node == in_node:
                    dep.index_transform = _ConcatIndexTransform(
                        offset=offsets[i:i+2], reverse=True)

            for dep in in_node.dependencies:
                if dep.broken_node == cat_node:
                    dep.index_transform = _ConcatIndexTransform(
                        offset=offsets[i:i+2], reverse=False)

    def _set_split_index_transform(self, split_node: Node):
        if split_node.type != OPTYPE.SPLIT:
            return

        chs = []
        for n in split_node.outputs:
            chs.append(_get_out_node_in_channels(n))

        offsets = [0]
        for ch in chs:
            offsets.append(offsets[-1]+ch)
        split_node.module.offsets = offsets
        for i, out_node in enumerate(split_node.outputs):
            for dep in split_node.dependencies:
                if dep.broken_node == out_node:
                    dep.index_transform = _SplitIndexTransform(
                        offset=offsets[i:i+2], reverse=False)

            for dep in out_node.dependencies:
                if dep.broken_node == split_node:
                    dep.index_transform = _SplitIndexTransform(
                        offset=offsets[i:i+2], reverse=True)


def _get_in_node_out_channels(node):
    ch = _get_node_out_channel(node)
    if ch is None:
        ch = 0
        for in_node in node.inputs:
            if node.type == OPTYPE.CONCAT:
                ch += _get_in_node_out_channels(in_node)
            else:
                ch = _get_in_node_out_channels(in_node)
    return ch


def _get_out_node_in_channels(node):
    ch = _get_node_in_channel(node)
    if ch is None:
        ch = 0
        for out_node in node.outputs:
            if node.type == OPTYPE.SPLIT:
                ch += _get_out_node_in_channels(out_node)
            else:
                ch = _get_out_node_in_channels(out_node)
    return ch
