#!/bin/env python

# usage: cfa.py sourcefile.txt outfile.mp3

# rule(frequency, amplitude, harmonic, loop=time|count, duration, fill=new|clone|quiet, a, d, s, r)
# ops: '':None, '=':assign, '+':increment, '*':scale

import sys, time, math, wave, random
from array import array  # twice as fast as plain lists

try:
    import pymedia.audio.sound as sound
    import pymedia.audio.acodec as acodec
    import pymedia.muxer as muxer
except:
    print "Please install PyMedia."
    print "(or fix wave_dump())"
    raise

pi = math.pi
sin = math.sin

format = sound.AFMT_S16_LE
sample_rate = 48000
channels = 1
snd = None
snd = sound.Output(sample_rate, channels, format)

def wave_dump(path, s):
    # doesn't depend on pymedia, but doesn't seem to write more than 1 second
    fw = wave.open(path, 'wb')
    fw.setparams((channels, 2, sample_rate, 0, 'NONE',''))
    fw.writeframes(s)
    fw.close()

def mp3_dump(path, s):
    # still no clue why this inserts zeros and halves playpack
    params = {'id': acodec.getCodecID('mp3'),
              'bitrate': 128000,
              'sample_rate': sample_rate,
              'channels': channels}
    mx   = muxer.Muxer('mp3')
    stId = mx.addStream(muxer.CODEC_TYPE_AUDIO, params)
    enc  = acodec.Encoder(params)
    fw   = open(path, 'wb')
    ss   = mx.start()
    fw.write(ss)
    enc_frames = enc.encode(s)
    for efr in enc_frames:
        ss = mx.write(stId, efr)
        if ss:
            fw.write(ss)
    if fw:
        if mx:
            ss = mx.end()
        if ss:
            fw.write(ss)
    fw.close()

def frange(start, stop, step):
    "floating range with adjustable step size"
    delta = stop - start
    step = float(step)
    fencepost = int(bool((delta/step)%1))
    return [start + i*step for i in range(0, int(delta/step) + fencepost)]

def frange2(start, stop, count):
    "floating range with adjustable number of elements"
    if count == 0:
        return []
    if stop == start:
        return [start] * count
    return frange(start, stop, (stop-start)/count)

def safe_zip(a, b):
    "unlike zip(), does not truncate to shortest"
    # not at all safe for infinite lists
    a_len = len(a)
    b_len = len(b)
    if a_len < b_len:
        a.extend([0] * (b_len-a_len))
    if a_len > b_len:
        b.extend([0] * (a_len-b_len))
    return zip(a,b)

def sin_builder(freq, amp=1.0, harmonic=1, phase=0, offset=0):
    "generates a sin wave one fundamental long"
    wavel = sample_rate / (freq*harmonic)
    if wavel < 1:
        return None  # check for this!
    phase = phase * pi / 180
    return array('f', [amp * (sin(x + phase) + offset) for x in frange2(0, 2*pi, wavel)] * harmonic)

def adsr(wave, a, d, s, r):
    "applies an envelope to a wave"
    assert a+d+r < 1.0
    l = len(wave)
    shape_a = frange2(0, 1.0, l*a)
    shape_d = frange2(1.0, s, l*d)
    shape_s = [s] * int(l * (1-a-d-r))
    shape_r = frange2(s, 0.0, l*r)
    shape = array('f', shape_a + shape_d + shape_s + shape_r)
    return array('f', [x*s for x,s in zip(wave, shape)])

def make_noise(wave, mp3=None, noisy=True):
    "plays/saves a wave"
    clip = 2**15 - 1
    w_peak = max(max(wave), abs(min(wave)))
    scale = clip / w_peak
    s = array('i', [int(i*scale) for i in wave])
    if noisy:
        snd.play(s)
    if mp3 is not None:
        mp3_dump(mp3, s)
        #wave_dump('cfa.wav', s)

def loadfile(path):
    "for samples"
    f = wave.open(path, 'rb' )
    # check format?
    return array('f', [int(i) for i in f.readframes()])

def inherit_arg(parent_tuple, child_tuple):
    "merge child and parent settings"
    parent_value = parent_tuple[1]
    op, value = child_tuple
    if value is None:
        return ('=', parent_value)
    if op not in ['=', '+', '*']:
        return ('=', parent_value)
    if op == '=':
        return ('=', value)
    if op == '+':
        return ('=', parent_value + value)
    if op == '*':
        return ('=', parent_value * value)

def repeat(sample, loop, duration,  fill):
    "not always safe, can not call() new random waveforms"
    def new_sample():
        if type(sample) == type(array('f', [])):
            return sample
        else:
            return None
    base = new_sample()
    reps = 1
    #if len(sample) == 0:
    #    return base
    if loop == 'time':
        reps = (duration - len(base)/float(sample_rate)) / (len(base)/float(sample_rate))
    if loop == 'count':
        reps = duration - 1
    if fill == 'clone':
        return base + base * int(reps) + base[0:int(len(base)*(reps%1))]
    if fill == 'quiet':
        return base + array('f', [0.0] * int(reps * len(base)))
    if fill == 'new':
        if loop == 'time':
            while len(base) < duration * sample_rate:
                base.extend(new_sample())
            return base[0:duration * sample_rate]
        if loop == 'count':
            for i in range(reps):
                base.extend(new_sample())
            return base

class Rule(object):
    "holds a mess of waveform settings"
    def __init__(self, **kwargs):  # name = ('op', value)
        stock = { 'name':('',None), 'path':('', None),
                  'frequency':('',0), 'amplitude':('',1.0), 'harmonic':('',0), 
                  'loop':('','time'), 'duration':('',1), 'fill':('','clone'), 
                  'a':('',0), 'd':('',0), 's':('',1), 'r':('',0) , 'cp':('',None)}
        for name in stock:
            setattr(self, name, stock[name])
        for name in kwargs:
            setattr(self, name, kwargs[name])
    def show(self):
        print self.name, self.frequency, self.amplitude, self.cp
    def wave(self):
        "returns the associated sin wave or sample wave"
        base = array('f', [])
        if self.name[1] == 'sin':
            base = sin_builder(self.frequency[1], amp=self.amplitude[1], harmonic=self.harmonic[1])
        if self.name[1] == 'path':
            base = loadfile(self.path[1])
        if self.name[1] == 'none':
            pass
        return base
    def recurse(self, sub_rule=None):
        "build a new Rule by combining parent and child"
        if sub_rule is None:
            return self
        #assert sub_rule.harmonic[1] > 0
        if (self.frequency[1] +  sub_rule.frequency[1]) * (self.harmonic[1] + sub_rule.harmonic[1]) > sample_rate:
            return None  # check for this!
        # has to be a better way to do this
        return Rule(name       = inherit_arg(self.name,      sub_rule.name),
                    path       = inherit_arg(self.path,      sub_rule.path),
                    frequency  = inherit_arg(self.frequency, sub_rule.frequency),
                    amplitude  = inherit_arg(self.amplitude, sub_rule.amplitude), 
                    harmonic   = inherit_arg(self.harmonic,  sub_rule.harmonic),
                    loop       = inherit_arg(self.loop,      sub_rule.loop),
                    duration   = inherit_arg(self.duration,  sub_rule.duration),
                    fill       = inherit_arg(self.fill,      sub_rule.fill),
                    #cp         = inherit_arg(self.cp,        sub_rule.cp),  inheriting cp yeilds infinite loops
                    a          = inherit_arg(self.a,         sub_rule.a),
                    d          = inherit_arg(self.d,         sub_rule.d),
                    s          = inherit_arg(self.s,         sub_rule.s),
                    r          = inherit_arg(self.r,         sub_rule.r))

class Rulebook:
    "hold a bunch of rules"
    def __init__(self):
        "set up stock system rules"
        self.book = {'sin':None, 'sample':None, 'none':None}
    def add(self, name, rule):
        if name in ['sin', 'sample', 'none']:
            return
        if name not in self.book:
            self.book[name] = []
        self.book[name].append(rule)
    def call(self, name, parent_rule=None):
        "creats a wave from the rule"
        assert name in self.book
        instructions = random.choice(self.book[name])
        waveform_p = array('f', [])
        waveform_s = array('f', [])
        subwave    = array('f', [])
        subwave_cp = array('f', [])
        for parallel_i in instructions:
            waveform_s = array('f', [])
            for serial_i in parallel_i:
                j = serial_i
                if parent_rule is not None:
                    j = parent_rule.recurse(serial_i)
                if j is None:
                    continue
                # conditions when repeat() is safe
                if j.fill[1] in ['clone', 'quiet'] or j.name[1] in ['sin', 'sample', 'none']:
                    if j.name[1] in ['sin', 'sample', 'none']:
                        subwave = j.wave()
                    else:
                        subwave = self.call(j.name[1], j)
                    if subwave is None:
                        continue
                    subwave = repeat(subwave, j.loop[1], j.duration[1], j.fill[1])
                else:  # not safe to use repeat()
                    subwave = self.call(j.name[1], j)
                    if subwave is None:
                        continue
                    if j.loop[1] == 'time':
                        while len(subwave) < j.duration[1] * sample_rate:
                            subwave.extend(self.call(j.name[1], j))
                    if j.loop[1] == 'count':
                        for i in range(j.duration[1] - 1):
                            subwave.extend(self.call(j.name[1], j))
                if j.cp[1] is not None:
                    subwave_cp = self.call(j.cp[1], j)
                    subwave = array('f', [a+b for a,b in safe_zip(subwave, subwave_cp)])
                subwave = adsr(subwave, j.a[1], j.d[1], j.s[1], j.r[1])
                waveform_s.extend(subwave)
            waveform_p = array('f', [a+b for a,b in safe_zip(waveform_s, waveform_p)])
        return waveform_p

# magical pyparsing BNF....
"""
variable name ::= "frequency" | "amplitude" | "harmonic" | "loop" | "duration" | "fill" | "a" | "d" | "s" | "r"
op_type ::= "=" | "+" | "*"
variable_value ::= number | "time" | "count" | "new" | "clone" | "quiet"
set_var ::= variable_name op_type variable_value
var_list ::= "" | set_var | set_var var_list
rule_contents ::= target_rule "(" var_list ")"
serial_sound ::= rule_contents rule_contents
parallel_sound ::= serial_sound | serial_sound "|" serial_sound | serial_sound "|" parallel_sound
rule ::= rule_name "=" parallel_sound
"""

def string_to_ast(s):
    from pyparsing import Word, OneOrMore, ZeroOrMore, Group,  Literal, Empty, alphas
    L = Literal
    word = Word(alphas + '_')
    number = Word('-1234567890.')
    #path = ???
    variable_name = (L('frequency') ^ L('amplitude') ^ L('harmonic') ^ L('loop') ^ L('duration') ^ L('fill') ^ L('path') ^ L('cp') ^ L('a') ^ L('d') ^ L('s') ^ L('r'))
    op_type = Word('=+*', exact=1)
    variable_value = (L('time') ^ L('count') ^ L('new') ^ L('clone') ^ L('quiet') ^ number ^ word)  # shoddy, no place for sample paths
    set_var = Group(variable_name + op_type + variable_value)
    target_rule = word
    rule_contents = Group(target_rule + L('(').suppress() + Group(ZeroOrMore(set_var)) + L(')').suppress())
    rule_serial = Group(OneOrMore(rule_contents))
    rule_name = word
    rule = (rule_name + L('=').suppress() + rule_serial) ^ ('|' + rule_serial)
    s2 = s.split('\n')
    ast = []
    for line in s2:
        line = line.strip()
        if line == '': continue
        try:
            ast.append(rule.parseString(line))
        except:
            print 'FAILED ON: ' + line
            raise
    return ast

def typify(s):
    "string to most restictive type"
    try:
        return int(s, 10)
    except:
        pass
    try:
        return float(s)
    except:
        pass
    return s

def ast_to_rulebook(ast):
    "a pain in the butt to debug"
    s_r = []
    p_r = []
    rb = Rulebook()
    old_rule_name = ''
    for rule_name, serial_rules in ast:
        s_r = []
        if rule_name != '|' and p_r != []:
            rb.add(old_rule_name, p_r)
            p_r = []
        for rule_call, settings in serial_rules:
            kwargs = {'name':('=',rule_call)}
            for var,op,value in settings:
                kwargs[var] = (op, typify(value))
            s_r.append(Rule(**kwargs))
        p_r.append(s_r)
        if rule_name != '|':
            old_rule_name = rule_name
    rb.add(old_rule_name, p_r)
    return rb

example_string = """
startsound = repeat()
repeat = tone(frequency=440 loop=count duration=3 fill=new) tone(frequency=660 loop=count duration=3 fill=new)
tone = no_shape(harmonic=1 loop=time duration=1 fill=clone a=0.1 d=0.1 s=0.5 r=0.1)
a = sin()
  | a(amplitude*0.5 harmonic+2)
a = sin()
  | a(amplitude*0.25 harmonic+4)
no_shape = a(a=0 d=0 s=1 r=0)
"""

ex_cps = """
startsound = c(duration=0.5 cp=organ) d(duration=0.5 cp=organ) e(duration=0.5 cp=organ)
organ = shaped(harmonic=1 cp=overtones)
overtones = tone()
          | overtones(amplitude*0.5 harmonic+4)
tone = sin(a=0 d=0 s=1 r=0)
shaped = none(a=0.1 d=0.1 s=0.5 r=0.1)
c = none(frequency=261.63)
d = none(frequency=293.66)
e = none(frequency=329.63)
f = none(frequency=349.23)
g = none(frequency=392.00)
a = none(frequency=440.00)
b = none(frequency=493.88)
"""


def example(a_string=None):
    if a_string is None:
        a_string = example_string
    print a_string + '\n\n'
    rb = ast_to_rulebook(string_to_ast(a_string))
    make_noise(rb.call('startsound'), mp3=None, noisy=True)

def main(argv=None):
    if argv is None:
        argv = sys.argv
    if len(argv) == 1:
        sys.exit(example())
    source_f = open(argv[1])
    target_f = argv[2]
    source_t = ''.join(source_f.readlines())
    rb = ast_to_rulebook(string_to_ast(source_t))
    make_noise(rb.call('startsound'), target_f)

if __name__ == "__main__":
    sys.exit(main())
    #example()
    #import cProfile
    #cProfile.run("example()", sort=1)
