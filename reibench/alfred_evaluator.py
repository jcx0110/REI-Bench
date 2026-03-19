import math
import os, json
import pprint
import random
import textwrap
import time, datetime
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '')
sys.path.insert(0, './alfred')

from PIL import Image, ImageDraw, ImageFont

from reibench.planners.alfred_planner import AlfredTaskPlanner
from reibench.envs.alfred.thor_connector import ThorConnector
from reibench.envs.alfred.utils import dotdict, load_task_json
from tqdm import tqdm

import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from reibench.evaluator import Evaluator

from reibench.utils.config_mapper import (
    get_planner_framework, get_data_type, get_data_types, get_prompting_method
)
import re
import time

font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf", 24)
log = logging.getLogger(__name__)
log_success = logging.getLogger(f"{__name__}_success")

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "planners", "prompts")


class AlfredEvaluator(Evaluator):
    def __init__(self, hparams):
        self.cfg = hparams
        method_config = get_prompting_method(hparams)
        self.COT = method_config.get('COT', False)
        self.TOCC = method_config.get('TOCC', False)
        self.planner_framework = get_planner_framework(hparams)
        self.split = self.cfg.split

    def evaluate(self):
        cfg = self.cfg

        global splits

        if self.planner_framework == "saycan":
            if len(cfg.planner.model_name) > 0:
                if not getattr(self, 'planner', None):
                    self.planner = AlfredTaskPlanner(cfg)
                    self.planner.reset()
            else:
                self.planner = None
        else:
            raise ValueError(f"Unknown planner_framework: {self.planner_framework}. Only 'saycan' is supported.")

        args_dict = {'data': 'data/raw/alfred/json_2.1.0', 'pframe': 300, 'fast_epoch': False,
                    'use_templated_goals': False, 'dout': 'exp/model', 'pp_folder': 'pp',
                    'reward_config': 'alfred/models/config/rewards.json', 'max_steps': 1000}
        
        splits = self.split

        with open(splits) as f:
            splits = json.load(f)

        number_of_dirs = len(list(os.listdir(args_dict['data'])))
        do_preprocessing = number_of_dirs < 50 
        if do_preprocessing:
            try:
                from alfred.data.preprocess import Dataset
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Preprocessing requested (data dirs < 50) but alfred.data.preprocess not found. "
                    "Add ALFRED data module (alfred/data/preprocess.py from askforalfred/alfred) or use a dataset with >= 50 task dirs to skip preprocessing."
                ) from None
            log.info("\nPreprocessing dataset... Do this once as required:")
            vocab = None
            dataset = Dataset(dotdict(args_dict), vocab)
            dataset.preprocess_splits(splits)

        assert cfg.alfred.eval_set in splits.keys()
        files = []

        for e in splits[cfg.alfred.eval_set]:
            if 'pick_two_obj_and_place' not in e['task']:
                files.append(e)

        if False: 
            for file in files:
                if 'trial_T20190907_012527_434405' in file['task']:
                    new_files = [file]
                    break
            files = new_files

        start = time.time()
        x_display = cfg.alfred.x_display
        save_path = cfg.out_dir
        results = self.evaluate_main(files, args_dict, self.planner, x_display, save_path)
        n = len(results)
        n_success = sum(1 for e in results if e['success'])
        latencies = [e['whole_plan_latency_ms'] for e in results if 'whole_plan_latency_ms' in e]
        log.info(f'success rate: {n_success / n * 100:.2f} % ({n_success}/{n})')
        log.info(f'elapsed: {str(datetime.timedelta(seconds=(time.time() - start)))}')
        if len(latencies) > 0:
            log.info(f'Average whole-plan latency: {sum(latencies) / len(latencies):.2f} ms')  

    def evaluate_main(self, tasks, args_dict, planner, x_display, save_path):
        results = []
        model_args = dotdict(args_dict)
        env = ThorConnector(x_display=x_display)

        if planner is not None:
            with open(os.path.join(save_path, 'prompt.txt'), 'w') as f:
                f.write(planner.prompt)

        train_gt_steps = None
        if self.cfg.alfred.eval_set == 'train':

            train_gt_steps = {}
            with open(self.cfg.prompt.example_file_path, 'r') as f:
                train_examples = json.load(f)
            for ex in train_examples:
                train_gt_steps[ex['task id']] = ex['NL steps']

        data_type = getattr(self.cfg, 'data_type', '')
        log.info(f"Evaluating {len(tasks)} tasks (data_type={data_type})")
        for i, task in enumerate(tqdm(tasks, leave=False, mininterval=10)):
            traj_data = load_task_json(task, args_dict["data"])
            r_idx = task['repeat_idx']
            try:
                if self.planner_framework == "saycan":
                    result = self.evaluate_task_saycan(env, traj_data, r_idx, model_args, planner, save_path, log_prompt=False, train_gt_steps=train_gt_steps)
                else:
                    raise ValueError(f"Unsupported planner_framework: {self.planner_framework}. Only 'saycan' is supported.")
                results.append(result)
                status = "success" if result['success'] else "fail"
                log.info(f"Task {i+1}/{len(tasks)}: {status}")
                if result['success']:
                    log_success.debug(f"{i+1}/{len(tasks)} {traj_data['root']}")

            except Exception as e:
                import traceback
                traceback.print_exc()
                log.info(f"Task {i+1}/{len(tasks)}: fail (exception: {repr(e)})")

        n_done = len(results)
        log.info(f"Finished {n_done}/{len(tasks)} tasks.")
        return results

    def instruction_organizing(self, traj_data, r_idx):
        data_type = get_data_type(self.cfg)
        ann = traj_data['turk_annotations']['anns'][r_idx]
        instruction_text = traj_data['turk_annotations']['anns'][r_idx][f'task_desc{data_type}']
        memory = ann.get(f'memory{data_type}', ann.get(f'robot_human_memory{data_type}', []))

        memory_instruction_text = ""
        for line in (memory if isinstance(memory, list) else [memory] if memory else []):
            memory_instruction_text = memory_instruction_text + "\n" + "Human previous inquiry(Not Required to Execute):" + line.replace("Human: ", "", 1)
        memory_instruction_text = memory_instruction_text + "\n" + "Human pending instruction:" + instruction_text.replace("Human: ", "", 1)
        return memory_instruction_text, instruction_text.replace("Human: ", "", 1)

    def evaluate_task_saycan(self, env, traj_data, r_idx, model_args, planner, save_path, log_prompt=False, train_gt_steps=None):
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        env.step(dict(traj_data['scene']['init_action']))
        env.set_task(traj_data, model_args, reward_type='dense')

        instruction_text, instruction_text_2 = self.instruction_organizing(traj_data, r_idx)
        
        done, success = False, False
        t = 0
        reward = 0
        imgs = [Image.fromarray(env.last_event.frame)]

        if self.cfg.planner.model_name.endswith('gpt-3.5-turbo') or 'gpt-4' in self.cfg.planner.model_name:
            step_by_step_mode = False
        else:
            step_by_step_mode = True

        if step_by_step_mode:
            goal_satisfied = False
            prev_steps = []
            prev_action_msg = []
            step_num = 0
            start = time.perf_counter()
            while not done:
                if self.cfg.alfred.eval_set == 'train' and train_gt_steps is not None:
                    if t >= len(train_gt_steps[traj_data['task_id']]):
                        step = 'done'
                    else:
                        step = train_gt_steps[traj_data['task_id']][t]
                    prompt = ''
                else:
                    step, prompt = planner.plan_step_by_step(instruction_text, prev_steps, prev_action_msg, None)
                    if step is None:
                        log.info("\tmax step reached")
                        break

                if log_prompt:
                    log.info(prompt)
                prev_steps.append(step)

                if step in ['done', 'done.', 'done.\n']:
                    done = True
                    prev_action_msg.append('')
                    break

                step_to_execute = step

                if step_to_execute in ['done', 'done.', 'done.\n']:
                    done = True
                    prev_action_msg.append('')
                    break
                step_num += 1
                try:
                    action_ret = env.llm_skill_interact(step_to_execute)
                    prev_action_msg.append(action_ret['message'])
                except Exception as e:
                    log.warning(e)
                    action_ret = {'success': False, 'message': str(e)}
                    prev_action_msg.append(action_ret['message'])
                imgs.append(env.write_step_on_img(self.cfg.planner.use_predefined_prompt, t+1, action_ret))

                if not action_ret['success']:
                    pass

                t_reward, t_done = env.get_transition_reward()
                reward += t_reward
                t += 1
            end = time.perf_counter()
            whole_latency_ms = (end - start) * 1000
        else:
            replan_times = 0
            goal_satisfied = False
            prev_steps = []
            prev_action_msg = []
            while not goal_satisfied and replan_times < 3 and goal_satisfied != "True":
                steps, prompt = planner.plan_whole(instruction_text, prev_steps, prev_action_msg)
                prev_steps = steps

                if log_prompt:
                    log.info(prompt)

                for si, step in enumerate(steps):
                    if step in ['done', 'done.', 'done.\n']:
                        done = True
                        break

                    step_to_execute = step
                    try:
                        action_ret = env.llm_skill_interact(step_to_execute)
                    except Exception as e:
                        log.warning(e)
                        action_ret = {'success': False, 'message': str(e)}
                    imgs.append(env.write_step_on_img(self.cfg.planner.use_predefined_prompt, t + 1, action_ret))
                    prev_action_msg.append(action_ret['message'])
                    if not action_ret['success']:
                        pass

                    t_reward, t_done = env.get_transition_reward()
                    reward += t_reward
                    t += 1
                goal_satisfied = env.get_goal_satisfied()
                replan_times += 1
        
        if step_by_step_mode and not 'DeepSeek-R1-Distill' in self.cfg.planner.model_name:
            goal_satisfied = env.get_goal_satisfied()

        goal_satisfied = env.get_goal_satisfied()
        if goal_satisfied:
            success = True
        log_entry = {'trial': traj_data['task_id'],
                    'scene': scene_name,
                    'type': traj_data['task_type'],
                    'repeat_idx': int(r_idx),
                    'goal_instr': instruction_text,
                    'inferred_steps': prev_steps,
                    'whole_plan_latency_ms': whole_latency_ms,
                    'success': success}
        return log_entry

    def save_result(self, result_dict, imgs, base_path='results'):
        if result_dict:
            filename = f"{result_dict['trial']}_{result_dict['repeat_idx']}"

            with open(os.path.join(base_path, filename + '.json'), "w") as outfile:
                json.dump(result_dict, outfile)
        else:
            filename = "images"

        widths, heights = zip(*(i.size for i in imgs))
        total_width = widths[0] * 5
        textbox_height = 70 
        total_height = math.ceil(len(imgs) / 5) * heights[0] + textbox_height
        new_im = Image.new('RGB', (total_width, total_height), color='white')

        if result_dict:
            text = 'Instruction: ' + result_dict['goal_instr']
            text_color = (0, 0, 0)  
            lines = textwrap.wrap(text, width=110)
            draw = ImageDraw.Draw(new_im)
            y_start = 10 if len(lines) > 1 else 35
            draw.multiline_text((10, y_start), '\n'.join(lines), font=font, fill=text_color)
            y_offset = textbox_height
        else:
            y_offset = 0

        x_offset = 0
        for im in imgs:
            new_im.paste(im, (x_offset, y_offset))
            x_offset += im.size[0]
            if x_offset >= total_width:
                x_offset = 0
                y_offset += im.size[1]

        success_str = 'success' if result_dict['success'] else 'fail'
        new_im.save(os.path.join(base_path, f"{filename}_{success_str}_{self.cfg.data_type}.png"))
